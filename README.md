# Forensic Face Restoration & Fusion API — Deployment Package

## What's in this folder

| File | Role |
|---|---|
| `serve_api.py` | The API your website calls. Run this. |
| `face_fusion_v2.py` | The fusion pipeline (landmarks, pose clustering, region blending). Imported by `serve_api.py`, not run directly in production. |
| `train_denoiser_model.py` | The NAFNet model definition + checkpoint loader. Imported by both of the above. |
| `denoiser_inference.pth` | Your actual trained model — 20.8M params, 45,000 training steps, 29.1dB validation PSNR. |
| `verify_setup.py` | **Run this first.** Local check, no server needed: packages import, torch/numpy actually interoperate, checkpoint loads, one image restores. Exits 1 with a specific fix on the first thing that's wrong. |
| `test_api.py` | **Run this after starting the server.** Hits `/health`, checks auth both ways on `/v1/model`, then a real `/v1/restore` call. |
| `requirements.txt` | Exact packages, version-tested together (see verification section below). |
| `.env.example` | Copy to `.env` and fill in before running. |
| `run.sh` | Loads `.env` and starts the server. |

---

## 1. Local setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If this machine has an NVIDIA GPU, install the matching CUDA build of torch from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) instead of the plain
`torch` from requirements.txt — pip's default gives you CPU-only, which works, just slower.

```bash
cp .env.example .env
# edit .env: set FORENSIC_API_KEYS to a real random string
```

```bash
./run.sh
# or directly:
python3 serve_api.py --host 0.0.0.0 --port 8000
```

You should see:
```
[engine] loaded /path/to/denoiser_inference.pth
         arch=nafnet params=20817699 step=45000 val_psnr=29.11... device=...
```
If `device=cuda` — GPU is being used. If `device=cpu` — it'll still work, just slower per request.

---

## 2. Phase 1 checklist — run these two scripts, in this order

```bash
# 1. Before starting anything — catches version/environment problems with a specific fix,
#    not a stack trace.
python verify_setup.py
```

You want to see `All checks passed.` at the bottom. If it fails, it tells you exactly which
package or version is wrong and the one command to fix it — this is what would have caught the
torch/numpy conflict below in about two seconds instead of a 500 error with no explanation.

```bash
# 2. Start the actual server (separate terminal, or background it)
python serve_api.py --host 0.0.0.0 --port 8000

# 3. In another terminal, with the server running:
python test_api.py --key YOUR_KEY
```

`test_api.py` generates its own test image by default, so it has zero setup — but you should
also run it against a real photo before trusting this:

```bash
python test_api.py --key YOUR_KEY --image path/to/a/real/blurry/photo.jpg
```

It writes the restored result to `test_api_restored.png` — open it and actually look at it.
A script returning HTTP 200 tells you the plumbing works; it does not tell you the model is any
good. Look at the picture.

## 2b. Test it locally with curl instead, if you'd rather

```bash
curl http://localhost:8000/health

curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/v1/model

curl -X POST http://localhost:8000/v1/restore \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@some_photo.jpg" \
  -o restored.png

curl -X POST http://localhost:8000/v1/reconstruct \
  -H "X-API-Key: YOUR_KEY" \
  -F "files=@photo1.jpg" -F "files=@photo2.jpg" -F "files=@photo3.jpg"
# -> {"job_id": "...", "poll": "/v1/jobs/..."}

curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/v1/jobs/THE_JOB_ID
```

`/v1/restore` answers immediately. `/v1/reconstruct` is a background job — poll `/v1/jobs/{id}`
until `status` is `"done"`, then the response includes links to `composite.png`,
`attribution_overlay.png`, and the full JSON report.

---

## 3. Connecting your website

**From your frontend (browser JS):**
```js
const res = await fetch("https://your-api-host:8000/v1/restore", {
  method: "POST",
  headers: { "X-API-Key": "YOUR_KEY" },
  body: (() => { const f = new FormData(); f.append("file", fileInput.files[0]); return f; })(),
});
const blob = await res.blob();
```

**Two things that will actually bite you in production, not hypothetical:**

1. **Mixed content.** If your website is `https://`, browsers will silently block requests to a
   plain `http://` API. You need TLS on this API too — put it behind a reverse proxy (Caddy is the
   least fiddly: one line of config gets you automatic HTTPS) or terminate TLS at your load
   balancer. Do not try to ship the API key check as your only security layer over plain HTTP.
2. **Don't put the API key in frontend JS.** Anything in browser code is visible to anyone who
   opens dev tools. Call this API from your backend (Node/Python/whatever your site runs), and
   have your own frontend talk to your own backend, which then holds the real key. If your
   "website" IS a static frontend with no backend, you need to add one for this — there's no way
   to keep a secret key safe in code the browser downloads.

**CORS:** set `FORENSIC_CORS_ORIGINS` in `.env` to your actual site's origin(s) before going live.
`*` is fine for local testing only.

---

## 4. What was actually verified before this was packaged (not just written)

I don't have your production GPU or your real footage, so I want to be precise about what "tested"
means here rather than let it sit as a vague claim:

- All three `.py` files compile with zero syntax errors.
- **Your actual checkpoint loads and runs.** I loaded `denoiser_inference.pth` through
  `train_denoiser_model.py`'s real loader, confirmed the parameter count matches exactly
  (20,817,699 both ways), and ran a real restoration forward pass end to end.
- **The full fusion pipeline runs end to end** — pose clustering, semantic region masks, per-region
  reliability scoring, Laplacian-pyramid blending, and JSON report generation — verified by
  substituting a synthetic-but-geometrically-real 468-point landmark set in place of live face
  detection (since I have no camera/photos here), so every downstream stage ran on real logic, not
  a mock. `serve_api.py`'s model loading was also verified directly against your real checkpoint.
- **I initially suspected a scoring bug** — that sensor noise could be scored as "sharper" than a
  clean image — and want to be upfront that on a fairer, more realistic test (structured content,
  not pure random noise standing in for a photo) the metric ranked correctly. My first test used
  an unrealistic stand-in for a "clean" image; correcting that changed the conclusion. No change
  was made to `face_fusion_v2.py` on the strength of a finding that didn't hold up.

**What is NOT verified:** real camera footage, real face detection accuracy (MediaPipe needs to
download a model file from Google's servers on first run — make sure this machine has outbound
internet access at least once), GPU behavior on your actual deployment hardware, and load/concurrency
behavior under real traffic. Test with real photos before pointing real users at this.

---

## 5. Second verification pass — a real bug was found and fixed here

Everything above was checked with syntax and imports. This pass actually started the server and
sent real HTTP requests, which caught something the first pass didn't:

**`/v1/restore` returned `500: RuntimeError: Numpy is not available`.** Cause: `requirements.txt`
allowed `torch>=2.2` (built against the NumPy 1.x C ABI) alongside `numpy<3` (which lets pip
install NumPy 2.x), and `opencv-python-headless`/`mediapipe` push NumPy toward 2.x on install. If
those resolve together, torch's C extension breaks the first time it touches a NumPy array — it
imports fine, and fails only when actually used, which is why the earlier "compiles with zero
syntax errors" check didn't catch it.

Fixed by pinning `torch==2.2.2` + `numpy>=1.26,<2` + `opencv-python-headless>=4.9,<4.11` together
(a loose `opencv-python-headless>=4.9` alone will happily resolve to a build that hard-requires
NumPy 2.x, undoing the pin above). Confirmed with `pip install --dry-run` that this exact
combination resolves with no conflicts, then confirmed for real: fresh install, server started,
`/health` → 200, `/v1/model` → 200 with your real checkpoint metadata, `/v1/restore` → 200 with an
actual denoised PNG back, on three different synthetic test frames.

If you ever change the torch or numpy pin in `requirements.txt`, re-run `verify_setup.py` before
trusting the result — this exact failure mode gives no warning at import time, only at first use.

**Visual sanity check, not just status codes:** the three `test_person_frame*.jpg` files shipped
alongside this package are synthetic degraded frames of one consistent (invented, non-photographic)
"identity" — used because a real face photo shouldn't sit in a shared package. Run them through
`test_api.py --image test_person_frame1.jpg` yourself and look at the output. In my own run, all
three came back visibly denoised: less grain, corrected brightness, cleaner edges — not just a
non-error response. Do the same with a couple of real photos before this touches real users.
