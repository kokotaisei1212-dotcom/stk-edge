"""Mask personal data the moment a conversation arrives. Nothing unmasked is stored or learned from.

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
]
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
            _NER = spacy.load("en_core_web_md", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
        except Exception:
            _NER = False
    return _NER


def _rules(text: str, known: list[str] | None) -> str:
    out = text
    for v in sorted(known or [], key=len, reverse=True):
        if v and len(v) >= 3:
            out = re.sub(re.escape(v), "<PII>", out, flags=re.I)
            for part in v.split():  # first or last name alone
                if len(part) >= 3 and part.isalpha():
                    out = re.sub(r"\b" + re.escape(part) + r"\b", "<PII>", out, flags=re.I)
    for rx, rep in _RULES:
        out = rx.sub(rep, out)
    return out


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


def _apply_ents(out: str, doc) -> str:
    for ent in reversed(doc.ents):
        if ent.label_ != "PERSON" or "<" in ent.text or not _looks_like_name(ent.text):
            continue
        if " " not in ent.text.strip() and _sentence_initial(out, ent.start_char):
            continue  # one capitalised word opening a sentence is usually just the sentence ("Gracias.", "Cancelé")
        out = out[:ent.start_char] + "<NAME>" + out[ent.end_char:]
    return out


def mask(text: str, known: list[str] | None = None, use_ner: bool = True, lang: str | None = None) -> str:
    """Mask personal data. `known` = values the caller already holds (the customer's name from the CRM); `lang` = the
    conversation's language when known (the entity model is English-only)."""
    return mask_many([text], [known], use_ner, [lang])[0]


def mask_many(texts: list[str], knowns: list[list[str] | None] | None = None, use_ner: bool = True,
              langs: list[str | None] | None = None) -> list[str]:
    """`mask` for a batch: the rules per text, then one pass of the entity model over the English texts of the batch."""
    knowns = knowns or [None] * len(texts)
    langs = langs or [None] * len(texts)
    outs = [_rules(t, k) for t, k in zip(texts, knowns)]
    nlp = _ner() if use_ner else False
    if not nlp or not outs:
        return outs
    idx = [i for i, (o, l) in enumerate(zip(outs, langs)) if _english_like(o, l)]
    for i, d in zip(idx, nlp.pipe([outs[i] for i in idx], batch_size=NER_BATCH)):
        outs[i] = _apply_ents(outs[i], d)
    return outs
