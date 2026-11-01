# STK edge

Run this next to your contact center platform and no conversation text leaves your network. It masks personal data
(emails, phone and card numbers, long ids, addresses, usernames, "my name is …", English person names, and any values
you pass in `known_pii`), turns the customer's first messages into hashed counts of word and character pieces, drops
the words, and forwards the request to STK.

```
docker run -p 8001:8001 -e STK_KEY=stk_... ghcr.io/kokotaisei1212-dotcom/stk-edge:latest
```

Then call `http://edge:8001` wherever the STK integration guide (https://stk-evidence.com/integrate) says
`https://api.stk-evidence.com`. Same requests, same answers.

What is rewritten: `turns` on `POST /v1/decide`, `/v1/conversations` and `/v1/history` become masked hashed features.
What is forwarded unchanged: everything else, including `POST /v1/ingest` (your own JSON): to keep words in your
cloud, send finished conversations as `/v1/conversations` through the shim, or without text at all.

The hashing is public (see `stk/features.py`): the numbers are not sentences, but someone holding them could test
whether a given word appears. That is why masking runs first.

This is the only part of STK that runs in your cloud. The STK engine is not included. See LICENSE.
