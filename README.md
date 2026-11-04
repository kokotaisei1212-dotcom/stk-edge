# STK edge

Run this next to your contact center platform and the conversation text does not leave your cloud. It masks personal
data (emails, phone and card numbers, long ids, addresses, usernames, person names checked in 15 languages (see
https://stk-evidence.com/docs#names), and any values you pass in `known_pii`, in every language), turns the customer's first messages into hashed counts of word and character pieces, drops
the words, and forwards the request to STK.

```
docker run -p 8001:8001 -e STK_KEY=stk_ai_... -e STK_FEATURE_KEY=<hex> ghcr.io/kokotaisei1212-dotcom/stk-edge:latest
# add -e STK_PII_MULTI=1 for the multilingual name model (about 280 MB more): European languages and Tagalog
```

`GET /health` on the container answers `{"ok": true, "version": ..., "stk_reachable": ...}` for your monitoring; it also
tells STK once a minute that it is alive (only its version), and STK emails you if it goes quiet for an hour.

Then call `http://edge:8001` wherever the STK integration guide (https://stk-evidence.com/integrate) says
`https://api.stk-evidence.com`. Same requests, same answers.

What is rewritten: `turns` on `POST /v1/decide`, `/v1/conversations` and `/v1/history` become masked hashed features.
What is forwarded unchanged: everything else. To keep words in your cloud, send finished conversations as
`/v1/conversations` through the shim, or without text at all.

The hashing itself is public (see `stk/features.py`), so the numbers alone could be tested against guessed words.
Two things stop that: masking runs first, and with `STK_FEATURE_KEY` (your center's own key: `GET
https://api.stk-evidence.com/v1/keys/feature` with your admin key) the positions are moved by your own secret
permutation before they leave. Without `STK_FEATURE_KEY`, STK applies the same key the moment the numbers arrive,
before anything is stored.

This is the only part of STK that runs in your cloud. The STK engine is not included. See LICENSE.
