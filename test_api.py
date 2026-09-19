import argparse
import io
import os
import sys
import time

GREEN, RED, YELLOW, RESET, BOLD = "\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m"


def ok(msg):
    print(f"  {GREEN}[OK]{RESET} {msg}")


def fail(msg, fix=None):
    print(f"  {RED}[FAIL]{RESET} {msg}")
    if fix:
        print(f"        {YELLOW}Fix:{RESET} {fix}")
    return False


def section(title):
    print(f"\n{BOLD}{title}{RESET}")


def make_test_image():
    """A synthetic degraded face-like image, generated on the fly so this script has
    no file dependency. Pass --image to test against a real photo instead."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(1)
    img = np.full((256, 256, 3), 150, np.uint8)
    cv2.circle(img, (128, 120), 80, (200, 190, 180), -1)
    cv2.circle(img, (100, 95), 10, (40, 40, 40), -1)
    cv2.circle(img, (156, 95), 10, (40, 40, 40), -1)
    cv2.ellipse(img, (128, 165), (30, 12), 0, 0, 180, (70, 40, 110), -1)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 15, img.shape), 0, 255).astype(np.uint8)
    ok_enc, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("FORENSIC_API_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--key", default=os.environ.get("FORENSIC_API_KEYS", "").split(",")[0])
    ap.add_argument("--image", default=None, help="Real image to restore. Default: generated.")
    ap.add_argument("--out", default="test_api_restored.png")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    passed = True

    try:
        import requests
    except ImportError:
        print(f"{RED}The 'requests' package is needed for this script (not for the API "
              f"itself).{RESET}\nInstall it with: pip install requests")
        return 1

    # ---- 1. /health — no key needed ----------------------------------------
    section(f"1. GET {base}/health")
    try:
        r = requests.get(f"{base}/health", timeout=10)
        if r.status_code != 200:
            passed = fail(f"HTTP {r.status_code}: {r.text[:200]}")
        else:
            data = r.json()
            ok(f"HTTP 200 — {data}")
            if not data.get("model_loaded"):
                passed = fail(
                    "Server is up but model_loaded=false — /v1/restore will return 503.",
                    "Check the server's own startup log for the '[engine] MODEL NOT LOADED' "
                    "line and the reason printed next to it.")
    except requests.exceptions.ConnectionError:
        return int(not fail(
            f"Could not connect to {base}",
            f"Is the server running? In another terminal: python serve_api.py\n"
            f"        Then re-run this script (or check --url if it's on a different host/port)."))
    except Exception as e:
        return int(not fail(f"{type(e).__name__}: {e}"))

    if not args.key:
        return int(not fail(
            "No API key given.",
            "python test_api.py --key YOUR_KEY\n"
            "        (must match one of the comma-separated values in FORENSIC_API_KEYS "
            "the server was started with — check the server's startup log if you didn't "
            "set one explicitly, it prints a generated throwaway key)."))

    # ---- 2. /v1/model — proves auth works both directions -------------------
    section(f"2. GET {base}/v1/model  (auth check)")
    try:
        r_noauth = requests.get(f"{base}/v1/model", timeout=10)
        if r_noauth.status_code not in (401, 403):
            passed = fail(f"No key sent, expected 401/403, got {r_noauth.status_code}. "
                          f"This endpoint may not be protecting itself correctly.")
        else:
            ok(f"No key -> HTTP {r_noauth.status_code} (correctly rejected)")

        r = requests.get(f"{base}/v1/model", headers={"X-API-Key": args.key}, timeout=10)
        if r.status_code == 403:
            return int(not fail(
                f"HTTP 403 with your key — the key is wrong.",
                f"Check it matches FORENSIC_API_KEYS on the server exactly (no quotes, no "
                f"trailing space)."))
        if r.status_code != 200:
            passed = fail(f"HTTP {r.status_code}: {r.text[:300]}")
        else:
            info = r.json()
            ok(f"Correct key -> HTTP 200 — arch={info.get('arch')} "
               f"step={info.get('step')} val_psnr={info.get('best_psnr')}")
    except Exception as e:
        passed = fail(f"{type(e).__name__}: {e}")

    # ---- 3. /v1/restore — the actual job ------------------------------------
    section(f"3. POST {base}/v1/restore")
    try:
        if args.image:
            if not os.path.isfile(args.image):
                return int(not fail(f"--image not found: {args.image}"))
            with open(args.image, "rb") as f:
                img_bytes = f.read()
            filename = os.path.basename(args.image)
        else:
            img_bytes = make_test_image()
            filename = "generated_test.jpg"
            print(f"        (no --image given — using a generated test image)")

        t0 = time.time()
        r = requests.post(
            f"{base}/v1/restore",
            headers={"X-API-Key": args.key},
            files={"file": (filename, io.BytesIO(img_bytes), "image/jpeg")},
            timeout=120)
        dt = time.time() - t0

        if r.status_code == 503:
            passed = fail(f"HTTP 503: {r.text[:300]}",
                          "The server is up but the model failed to load. Run "
                          "verify_setup.py on the server machine to see why.")
        elif r.status_code != 200:
            passed = fail(f"HTTP {r.status_code}: {r.text[:300]}")
        elif r.headers.get("content-type") != "image/png":
            passed = fail(f"Expected image/png back, got "
                          f"{r.headers.get('content-type')}: {r.content[:200]}")
        else:
            with open(args.out, "wb") as f:
                f.write(r.content)
            ok(f"HTTP 200 in {dt:.2f}s — {len(r.content)/1024:.1f} KB PNG")
            ok(f"input size:   {r.headers.get('x-input-size', '?')}")
            ok(f"model arch:   {r.headers.get('x-model-arch', '?')}")
            ok(f"checkpoint:   {r.headers.get('x-model-sha256', '?')}")
            print(f"        Wrote {args.out} — open it and look at it.")
    except requests.exceptions.Timeout:
        passed = fail("Request timed out after 120s.",
                      "Normal on a slow CPU for a large image. Try a smaller --image, or "
                      "check the server has a GPU available.")
    except Exception as e:
        passed = fail(f"{type(e).__name__}: {e}")

    # ---- summary --------------------------------------------------------------
    section("Summary")
    if passed:
        print(f"  {GREEN}{BOLD}All checks passed.{RESET} The API is reachable, authenticated, "
              f"and restoring images.")
        return 0
    else:
        print(f"  {RED}{BOLD}Fix the FAIL items above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
