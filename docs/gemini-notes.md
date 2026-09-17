# Gemini: what works, what breaks (verified 2026-09-16)

Proving the last external dependency before the Gemini build block. **No product code
is built on Gemini yet** — this is a capability check plus the disk cache every future
call must go through (`utils/llm_cache.py`).

## Model availability

`gemini-2.0-flash` and `gemini-2.5-flash` are **gone** for this key:

> 404 — "This model models/gemini-2.0-flash is no longer available. Please update your
> code to use models/gemini-3.6-flash for the latest features and improvements."

`ListModels` returns 41 models supporting `generateContent`. **Use `gemini-3.6-flash`**
(1M in / 65k out). `gemini-flash-latest` also resolves, but a floating alias can change
under a recorded demo, so the cache pins an explicit model name.

## Does it return parseable JSON?

| Mode | Fenced in markdown? | Parseable | Verdict |
|---|---|---|---|
| Plain prompt ("return ONLY JSON") | **Always** (3 of 3) | Yes, after stripping fences | Needs a strip step |
| `responseMimeType: application/json` + `responseSchema` | **Never** (4 of 4) | Yes, directly | **Use this** |

Schema mode is the one to build on: raw `json.loads` works, the enum is respected, and
every required key came back. `GeminiResponse.parsed_json()` strips fences anyway, so a
plain-prompt fallback still parses.

## Failure modes to build around

1. **It does not refuse or abstain.** Given a *flat grey 512×512 square* with no horizon,
   no landmarks and no sky, it answered `haze_severity: 90, visibility_km_estimate: 0.5,
   confidence: 0.4` and reasoned "completely featureless grey background". Confidence
   only fell to 0.4, not to zero.
   **Guardrail needed:** Gemini's own confidence cannot be the only filter. A citizen
   photo should also be checked against the hex's satellite AAI / NO2 and the nearest
   working monitor, and disagreement should downweight or reject the reading. Require a
   visible horizon or sky region, and treat any single photo as one noisy vote.
2. **503 "model is currently experiencing high demand"** hit twice in ~10 calls. Transient;
   retry with backoff (the wrapper does).
3. **429 quota** (see limits). The wrapper honours the server's `retryDelay`.
4. **Values wobble between identical calls.** The same hazy photo returned haze 75 / 75 / 80
   and visibility 4.5 / 3.5 / 4.0 km across runs. Source and severity band are stable;
   treat the numbers as ±10, not as precise readings. Caching also makes the demo
   deterministic, which matters more than the wobble.
5. **"Thinking" is on by default** and dominates token use: 155 thought tokens to answer
   "hi". `thinkingConfig: {thinkingBudget: 0}` works on 3.6-flash, cuts latency from
   7.4 s to 5.2 s, and did not degrade the JSON.

## Rate limits (observed, not from docs)

Burst of tiny calls until refusal:

```
quotaMetric: generativelanguage.googleapis.com/generate_content_free_tier_requests
quotaId:     GenerateRequestsPerMinutePerProjectPerModel-FreeTier
quotaValue:  5
retryDelay:  6s
```

- **5 requests per minute, per project per model** on the free tier. 6 calls landed in 12 s
  (sliding window) before the 7th was refused; the window reopened within ~10 s.
- **20 requests per day, per project per model** (observed 2026-09-17, after the note above said
  no daily cap had been seen):

  ```
  quotaId:    GenerateRequestsPerDayPerProjectPerModel-FreeTier
  quotaValue: 20
  ```

  The wrapper's retries cannot outwait it: the daily window resets at midnight Pacific
  (~12:30 IST), so a 429 on the daily quota must stop a batch, not retry.

**Consequences for the build:**

- One call per hex does not fit a day's budget. Source attribution and alerts are **batched: one
  call per date for all demo hexes** (6 calls for 3 dates), each hex analysed independently inside
  the batch and matched back by `hex_id`, with missing or extra ids flagged.
- Calls on 2026-09-17: 17 for the photo corpus (6 wasted on a prompt that omitted the 0-100
  scale, 1 timeout), 2 single-hex attributions later superseded by the batched prompt, then the
  daily cap. All Gemini features stay on `gemini-3.6-flash`; the batched runs happen after the
  next reset.
- Latency: 1.1–12.7 s per call (median ~5 s); image calls are not noticeably slower than text.

**Implication for the demo:** 5 RPM makes a live multi-photo demo fragile. Everything
replays from cache.

## The cache (`utils/llm_cache.py`)

Every Gemini call goes through it. No exceptions.

```python
from utils.llm_cache import gemini, text_part, image_part
resp = gemini([text_part(PROMPT), image_part("photo.jpg")], prompt_version="haze-v1",
              generation_config={"responseMimeType": "application/json", "responseSchema": SCHEMA})
resp.parsed_json(), resp.cached
```

- **Key** = sha256 of model + `prompt_version` + endpoint + request body, with image bytes
  folded in as their own digest. Verified: the key changes with the image, changes with the
  prompt version, and is stable on repeat.
- **Replay** is a file read: no network, no API key needed. Verified identical text on replay.
- `--no-cache` (or `use_cache=False`) forces a fresh call and overwrites the entry.
- `--offline` / `AEROGAP_GEMINI_OFFLINE=1` refuses to touch the network and raises
  `OfflineCacheMiss` on a miss. **Set this for the recorded video and the stage demo.**
- Cache lives in `data/gemini_cache/`, which is gitignored — the demo machine must carry it.
- `python -m utils.llm_cache --stats | --verify | --prune-errors`.

## Reproducing

```bash
python -m utils.llm_cache --stats
# test images: data/gemini_test/{hazy,clear,ambiguous}.jpg
```

Test images: Boston skyline haze (public domain, DPLA via Wikimedia Commons), Jakarta
skyline at blue hour (CC BY-SA 4.0), plus a synthetic grey square as the ambiguous control.
