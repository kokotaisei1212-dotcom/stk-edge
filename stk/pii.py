"""Mask personal data the moment a conversation arrives, before it is hashed; the text is not stored.

Two layers: patterns (emails, cards, phones, long IDs, addresses, usernames, stated names) and, when the
language model is installed, named-entity detection for person names anywhere in the text. Batches go through
`mask_many`, which runs the entity model over all texts at once (2-3x faster than one call per text)."""
import os, re

_RULES = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<CARD>"),
    (re.compile(r"\(?\+?\d{1,3}\)?[ -.]?\(?\d{2,4}\)?[ -.]?\d{3,4}[ -.]?\d{3,4}\b"), "<PHONE>"),
    (re.compile(r"\b\d{6,}\b"), "<NUMBER>"),
    (re.compile(r"\b\d{1,5}\s+[A-Za-z0-9 ]{2,30}\b(?:st|street|ave|avenue|rd|road|blvd|lane|ln|dr|drive)\b", re.I), "<ADDRESS>"),
    (re.compile(r"(?i)\b(my name is|name's|this is|i am|i'm)\s+([A-Z][a-z]+)(\s+[A-Z][a-z]+)?"), r"\1 <NAME>"),
    (re.compile(r"(?i)\b(username|user name|user id|login|account name)\s*(?:is|:)?\s*[\w.-]+"), r"\1 <USERNAME>"),
    # titles and "spoke to <name>": names the entity model can miss ("Kenji Watanabe")
    (re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr|Mx)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"), r"\1 <NAME>"),
    (re.compile(r"(?i)\b(spoke|speak|speaking|talked|talk|talking|chatted)\s+(to|with)\s+(?-i:[A-Z][a-z]+)(?:\s+(?-i:[A-Z][a-z]+))?"), r"\1 \2 <NAME>"),
]

# Japanese names: no entity model (the Japanese ones need a large dictionary). Three patterns: a name with an
# honorific (田中様, 佐藤さん), a name introduced (田中太郎と申します, 名前は鈴木です), and a common family name with
# a given name after it (高橋健一). Words that only look like a name before 様/さん (お客様, 担当者様) are left alone.
_JK = r"[一-龥々]"
_JN = rf"(?:{_JK}{{1,4}}(?:\s?{_JK}{{1,4}})?|[ァ-ヶー]{{2,8}}(?:[\s・]?[ァ-ヶー]{{2,8}})?)"
_JA_NOT = {"客", "お客", "皆", "皆様", "神", "奥", "旦那", "担当", "担当者", "店長", "会員", "利用者", "各位", "医者", "社長",
           "部長", "課長", "先生", "王", "殿", "お子", "子供", "家族", "相手", "業者", "配達員", "運転手", "皆さん"}
_JA_HON = re.compile(rf"({_JN})\s?(様|さま|さん|くん|君|殿|氏|ちゃん)")
_JA_INTRO = re.compile(rf"(名前は|氏名は|私は|わたしは|わたくしは)?\s*({_JN})\s*(と申します|といいます|と言います|と申す者)")
_JA_LABEL = re.compile(rf"(名前|氏名|お名前)(は|:|：)\s*({_JN})")
_JA_SURNAMES = ("佐藤 鈴木 高橋 田中 伊藤 渡辺 山本 中村 小林 加藤 吉田 山田 佐々木 山口 松本 井上 木村 斎藤 清水 山崎 池田 橋本 "
                "阿部 石川 山下 中島 石井 小川 前田 岡田 長谷川 藤田 後藤 近藤 村上 遠藤 青木 坂本 斉藤 福田 太田 西村 藤井 金子 "
                "岡本 藤原 中野 三浦 原田 中川 松田 竹内 小野 田村 中山 和田 石田 森田 上田 柴田 酒井 工藤 横山 宮崎 宮本 内田 "
                "高木 安藤 島田 谷口 大野 丸山 今井 高田 藤本 武田 村田 上野 杉山 増田 小山 大塚 平野 菅原 久保 松井 千葉 岩崎 "
                "桜井 木下 野口 松尾 菊地 野村 新井 渡部 井澤 伊沢 小松 大西 吉川 山内").split()
_JA_FULL = re.compile("(" + "|".join(sorted(_JA_SURNAMES, key=len, reverse=True)) + rf")\s?({_JK}{{1,3}})?")


# ---- names in other languages, by pattern ------------------------------------------------------------------------
# Latin-script languages: a cue (an introduction, a title, Tagalog's "si") and the capitalised words right after it.
# Capitals are checked with str.isupper, so accented and Vietnamese capitals count; a lower-case word after a cue
# ("soy cliente", "si customer") is a word, not a name, and stays.
_LATIN_CUES = [
    # Spanish
    r"me llamo", r"mi nombre es", r"soy", r"habl[éeo] con", r"sr\.", r"sra\.", r"srta\.", r"señor", r"señora", r"señorita", r"don", r"doña",
    # Portuguese
    r"meu nome é", r"me chamo", r"eu sou", r"sou o", r"sou a", r"falei com", r"senhor", r"senhora", r"dona",
    # French
    r"je m['’]appelle", r"mon nom est", r"je suis", r"j['’]ai parlé (?:avec|à)", r"m\.", r"mme", r"mlle", r"monsieur", r"madame", r"mademoiselle",
    # German, Italian
    r"ich hei(?:ß|ss)e", r"mein name ist", r"herr", r"frau", r"mi chiamo", r"il mio nome è", r"sono", r"ho parlato con", r"sig\.", r"sig\.ra", r"signor", r"signora",
    # Tagalog
    r"ako si", r"si", r"kay", r"ginoong", r"ginang", r"binibining", r"g\.", r"gng\.", r"bb\.",
    # Indonesian / Malay
    r"nama saya", r"bapak", r"pak", r"ibu", r"saudara", r"sdr\.",
    # Vietnamese
    r"tôi là", r"tên tôi là", r"tên là", r"anh", r"chị", r"ông", r"bà",
    # English titles and introductions the entity model can miss
    r"my name is", r"this is",
]
_LATIN_CUE = re.compile(r"(?i)(?<![\w])(" + "|".join(_LATIN_CUES) + r")\s+")
_PART = {"de", "da", "do", "dos", "das", "del", "dela", "la", "van", "von", "y", "e", "bin", "binti", "al"}
_TOK = re.compile(r"[^\s,.;:!?¿¡()\"“”]+")


def _latin(out: str) -> str:
    spans, pos = [], 0
    for m in _LATIN_CUE.finditer(out):
        if m.start() < pos:
            continue
        i, taken, end = m.end(), 0, None
        while taken < 4:
            t = _TOK.match(out, i)
            if not t:
                break
            w = t.group(0)
            if w[:1].isupper() and not w.isupper() or (w.isupper() and len(w) > 1 and taken):
                end, taken = t.end(), taken + 1
            elif w.lower() in _PART and taken:
                pass
            else:
                break
            nxt = re.match(r"\s+", out[t.end():])
            if not nxt:
                break
            i = t.end() + nxt.end()
        if end:
            spans.append((m.end(), end)); pos = end
    for a, b in reversed(spans):
        out = out[:a] + "<NAME>" + out[b:]
    return out


# Arabic: after "my name is", a title or "I spoke with", one or two words; a word with the article (ال...) is a noun.
_AR = re.compile(r"(اسمي|السيد|السيدة|الأستاذ|الأستاذة|الآنسة|تحدثت مع|كلمت)\s+((?!ال)[\u0621-\u064A]+)(\s+(?!ال)[\u0621-\u064A]+)?")
# Hindi: "my name is ... hai", a title before, or "ji" after a name (not after yes/no/you).
_HI_NAME = re.compile(r"(मेरा नाम|मेरा नाम है)\s+((?:[\u0900-\u097F]+\s+){0,2}?[\u0900-\u097F]+)(\s+है)")
_HI_TITLE = re.compile(r"(श्री|श्रीमती|सुश्री|कुमारी)\s+([\u0900-\u097F]+)(\s+[\u0900-\u097F]+)?")
_HI_JI = re.compile(r"([\u0900-\u097F]+)\s?(जी)(?![\u0900-\u097F])")
_HI_NOT = {"हाँ", "हां", "नहीं", "आप", "जी", "ठीक", "धन्यवाद", "ग्राहक", "सर", "मैडम"}
# Korean: an introduction, or a name before 씨/님 (not 고객님 and other roles).
_KO_INTRO = re.compile(r"(저는|제 이름은|이름은)\s?([가-힣]{2,4}?)(입니다|이에요|예요|이고요|라고 합니다|이라고 합니다)")
_KO_HON = re.compile(r"([가-힣]{2,4})\s?(씨|님)")
_KO_NOT = {"고객", "상담사", "상담원", "기사", "선생", "사장", "회원", "담당자", "부장", "과장", "팀장", "직원", "어머", "아버", "여러분", "손", "관리자", "택배기사"}
# Chinese: an introduction, or a family name before a title.
_ZH_SURNAMES = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
_ZH_INTRO = re.compile(r"(我叫|我的名字是|我的名字叫|我姓)([\u4e00-\u9fff]{1,3})")
_ZH_HON = re.compile(rf"([{_ZH_SURNAMES}][\u4e00-\u9fff]{{0,2}}?)(先生|女士|小姐|老师|经理|太太)")
# Thai: "my name is" (ผมชื่อ, ฉันชื่อ...) and the name and family name after it; titles นาย/นาง/นางสาว before a name.
_TH = re.compile(r"((?:ผม|ฉัน|ดิฉัน|หนู|เรา)?ชื่อ(?:ว่า)?)\s?([\u0E00-\u0E7F]{2,20})(\s[\u0E00-\u0E7F]{2,12})?(?=\s|$|ครับ|ค่ะ|คะ)")
_TH_TITLE = re.compile(r"(นางสาว|นาย|นาง)([\u0E00-\u0E7F]{2,20})(?=\s|$)")
_TH_NOT = ("จ้าง", "หน้า", "ทุน", "อำเภอ", "ตำรวจ", "กอง", "พยาบาล", "แบบ", "งาน")  # employer, broker, nurse... words, not names


def _other_scripts(out: str) -> str:
    if re.search(r"[\u0600-\u06FF]", out):
        out = _AR.sub(lambda m: m.group(1) + " <NAME>", out)
    if re.search(r"[\u0900-\u097F]", out):
        out = _HI_NAME.sub(lambda m: m.group(1) + " <NAME>" + m.group(3), out)
        out = _HI_TITLE.sub(lambda m: m.group(1) + " <NAME>", out)
        out = _HI_JI.sub(lambda m: m.group(0) if m.group(1) in _HI_NOT else "<NAME> " + m.group(2), out)
    if re.search(r"[가-힣]", out):
        out = _KO_INTRO.sub(lambda m: m.group(1) + " <NAME>" + m.group(3), out)
        out = _KO_HON.sub(lambda m: m.group(0) if m.group(1) in _KO_NOT or any(m.group(1).endswith(x) for x in _KO_NOT) else "<NAME>" + m.group(2), out)
    if re.search(r"[\u4e00-\u9fff]", out) and not re.search(r"[ぁ-んァ-ヶ]", out):  # Chinese, not Japanese (kana)
        out = _ZH_INTRO.sub(lambda m: m.group(1) + "<NAME>", out)
        out = _ZH_HON.sub(lambda m: "<NAME>" + m.group(2), out)
    if re.search(r"[\u0E00-\u0E7F]", out):
        out = _TH.sub(lambda m: m.group(1) + " <NAME>", out)
        out = _TH_TITLE.sub(lambda m: m.group(0) if m.group(2).startswith(_TH_NOT) else m.group(1) + "<NAME>", out)
    return out


def _ja(out: str) -> str:
    if not re.search(r"[ぁ-んァ-ヶ一-龥]", out):
        return out
    out = _JA_INTRO.sub(lambda m: (m.group(1) or "") + "<NAME>" + m.group(3), out)
    out = _JA_LABEL.sub(lambda m: m.group(1) + m.group(2) + "<NAME>", out)
    out = _JA_HON.sub(lambda m: m.group(0) if m.group(1).replace(" ", "") in _JA_NOT or m.group(1).startswith("お") else "<NAME>" + m.group(2), out)
    return _JA_FULL.sub("<NAME>", out)
_NER = None
NER_BATCH = 64
_LATIN = re.compile(r"[A-Za-z]")
_LETTER = re.compile(r"[^\W\d_]")
_PARTICLES = {"de", "del", "van", "von", "da", "di", "la", "le", "du", "bin", "al", "der", "den", "y", "e"}


def _ner():
    global _NER
    if _NER is None:
        if os.environ.get("STK_PII_NER", "1") == "0":
            _NER = False
            return _NER
        try:
            import spacy
            # the small English model: about 90 MB in memory (the medium one takes about 300 MB, too much next to the
            # engine on a 512 MB server); the patterns above catch what it misses
            for name in ("en_core_web_sm", "en_core_web_md"):
                try:
                    _NER = spacy.load(name, disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
                    break
                except OSError:
                    continue
            else:
                _NER = False
        except Exception:
            _NER = False
    return _NER


def _rules(text: str, known: list[str] | None) -> str:
    out = text
    for v in sorted(known or [], key=len, reverse=True):
        if v and (len(v) >= 3 or (len(v) == 2 and not _LATIN.search(v))):  # two-character names (赵敏, 王伟) count
            out = re.sub(re.escape(v), "<PII>", out, flags=re.I)
            for part in v.split():  # first or last name alone
                if len(part) >= 3 and part.isalpha():
                    out = re.sub(r"\b" + re.escape(part) + r"\b", "<PII>", out, flags=re.I)
    for rx, rep in _RULES:
        out = rx.sub(rep, out)
    return _other_scripts(_latin(_ja(out)))


def _english_like(text: str, lang: str | None) -> bool:
    """The entity model knows English. On other languages it tags whole sentences as names and destroys the signal
    (found in the 30-center world: 21 of 60 Japanese and 47 of 60 Spanish messages mangled), so it runs only on text
    declared English (or undeclared) and written mostly in Latin letters."""
    if lang and not str(lang).lower().startswith("en"):
        return False
    letters = _LETTER.findall(text)
    return not letters or len(_LATIN.findall(text)) / len(letters) >= 0.7


def _looks_like_name(s: str) -> bool:
    toks = s.replace("-", " ").split()
    return 1 <= len(toks) <= 4 and all(t[:1].isupper() or t.lower() in _PARTICLES for t in toks) and any(t[:1].isupper() for t in toks)


def _sentence_initial(text: str, at: int) -> bool:
    before = text[:at].rstrip()
    return not before or before[-1] in ".!?\n"


def _apply_ents(out: str, doc, label: str = "PERSON") -> str:
    for ent in reversed(doc.ents):
        if ent.label_ != label or "<" in ent.text or not _looks_like_name(ent.text):
            continue
        if " " not in ent.text.strip() and _sentence_initial(out, ent.start_char):
            continue  # one capitalised word opening a sentence is usually just the sentence ("Gracias.", "Cancelé")
        out = out[:ent.start_char] + "<NAME>" + out[ent.end_char:]
    return out


def mask(text: str, known: list[str] | None = None, use_ner: bool = True, lang: str | None = None) -> str:
    """Mask personal data. `known` = values the caller already holds (the customer's name from the CRM); `lang` = the
    conversation's language when known (the entity model is English-only)."""
    return mask_many([text], [known], use_ner, [lang])[0]


_XX = None
_XX_LANGS = {"es", "pt", "fr", "de", "it", "tl", "nl", "ca", "ro"}  # where the multilingual model was checked to find names


def _xx():
    """The multilingual entity model (xx_ent_wiki_sm), only with STK_PII_MULTI=1: it adds about 280 MB, which does not
    fit next to the engine on a 512 MB server, but does on stk-edge or a larger server. Off: the patterns alone."""
    global _XX
    if _XX is None:
        _XX = False
        if os.environ.get("STK_PII_MULTI", "0") == "1":
            try:
                import spacy
                _XX = spacy.load("xx_ent_wiki_sm")
            except Exception:
                _XX = False
    return _XX


def mask_many(texts: list[str], knowns: list[list[str] | None] | None = None, use_ner: bool = True,
              langs: list[str | None] | None = None) -> list[str]:
    """`mask` for a batch: the rules per text, then one pass of the entity model over the English texts of the batch."""
    knowns = knowns or [None] * len(texts)
    langs = langs or [None] * len(texts)
    outs = [_rules(t, k) for t, k in zip(texts, knowns)]
    nlp = _ner() if use_ner else False
    if not outs:
        return outs
    if not nlp:
        nlp = None
    idx = [i for i, (o, l) in enumerate(zip(outs, langs)) if _english_like(o, l)] if nlp else []
    for i, d in zip(idx, nlp.pipe([outs[i] for i in idx], batch_size=NER_BATCH) if nlp else []):
        outs[i] = _apply_ents(outs[i], d)
    xx = _xx() if use_ner else False
    if xx:
        mi = [i for i, l in enumerate(langs) if l and str(l).lower()[:2] in _XX_LANGS]
        for i, d in zip(mi, xx.pipe([outs[i] for i in mi], batch_size=NER_BATCH)):
            outs[i] = _apply_ents(outs[i], d, label="PER")
    return outs
