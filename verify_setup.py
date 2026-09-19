import argparse
import os
import sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FORENSIC_MODEL",
                    "denoiser_inference.pth"))
    ap.add_argument("--image", default=None,
                    help="Optional real image to restore. Default: a generated test image.")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    passed = True

    # ---- 1. Python version -------------------------------------------------
    section("1. Python version")
    if sys.version_info < (3, 9):
        passed = fail(f"Python {sys.version.split()[0]} — need 3.9+",
                      "Install Python 3.9 or newer and recreate the virtual environment.")
    else:
        ok(f"Python {sys.version.split()[0]}")

    # ---- 2. Package imports --------------------------------------------------
    section("2. Required packages")
    mods = ["numpy", "cv2", "torch", "mediapipe", "fastapi", "uvicorn"]
    versions = {}
    for m in mods:
        try:
            mod = __import__(m)
            v = getattr(mod, "__version__", "?")
            versions[m] = v
            ok(f"{m} {v}")
        except ImportError as e:
            passed = fail(f"{m} not importable: {e}",
                          "pip install -r requirements.txt  (inside your virtual env)")

    if not passed:
        print(f"\n{RED}Stopping here — fix the imports above before continuing.{RESET}")
        return 1

    # ---- 3. torch / numpy interop -------------------------------------------
    # This is the exact failure this deployment hit: torch built against the numpy
    # 1.x ABI raises "RuntimeError: Numpy is not available" the first time it touches
    # a numpy array, if numpy 2.x is installed alongside it. Import succeeding above
    # does NOT catch this — it only shows up here.
    section("3. torch / numpy interoperability")
    try:
        import numpy as np
        import torch
        t = torch.from_numpy(np.zeros((2, 2), dtype=np.float32))
        _ = t + 1
        ok(f"torch.from_numpy works (torch {torch.__version__}, numpy {np.__version__})")
    except Exception as e:
        passed = fail(
            f"{type(e).__name__}: {e}",
            "This is a version mismatch, not a code bug. Run:\n"
            "              pip install \"torch==2.2.2\" \"numpy<2\"\n"
            "        then re-run this script. See requirements.txt for why.")

    try:
        import cv2
        _ = cv2.GaussianBlur(np.zeros((8, 8, 3), np.uint8), (3, 3), 0)
        ok(f"opencv working (cv2 {cv2.__version__})")
    except Exception as e:
        passed = fail(f"opencv failed: {type(e).__name__}: {e}",
                      "pip install \"opencv-python-headless>=4.9,<4.11\"")

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            ok("No CUDA GPU detected — will run on CPU (slower, still correct)")
    except Exception:
        pass

    if not passed:
        print(f"\n{RED}Stopping here — fix the above before continuing.{RESET}")
        return 1

    # ---- 4. Model file + load ------------------------------------------------
    section("4. Model checkpoint")
    model_path = args.model
    if not os.path.isabs(model_path):
        for candidate in (model_path, os.path.join(here, model_path)):
            if os.path.isfile(candidate):
                model_path = candidate
                break

    if not os.path.isfile(model_path):
        return int(not fail(
            f"Not found: {model_path}",
            "Pass the real path: python verify_setup.py --model /path/to/denoiser_inference.pth\n"
            "        or set FORENSIC_MODEL, or put the file next to this script."))

    size_mb = os.path.getsize(model_path) / 1e6
    ok(f"Found {model_path} ({size_mb:.1f} MB)")

    sys.path.insert(0, here)
    try:
        import time
        from train_denoiser_model import load_denoiser
        t0 = time.time()
        net = load_denoiser(model_path, device=device, prefer_ema=True)
        params = sum(p.numel() for p in net.parameters())
        meta = getattr(net, "meta", {}) or {}
        ok(f"Loaded in {time.time()-t0:.1f}s — {params:,} params, "
           f"arch={meta.get('arch','?')}, step={meta.get('step','?')}, "
           f"val_psnr={meta.get('best_psnr','?')}")
    except Exception as e:
        passed = fail(f"{type(e).__name__}: {e}",
                      "Make sure train_denoiser_model.py is in the same folder as this "
                      "script and matches the checkpoint (same file the model was exported "
                      "with).")
        print(f"\n{RED}Stopping here.{RESET}")
        return 1

    # ---- 5. Actually restore one image ---------------------------------------
    section("5. Restore one image")
    try:
        from train_denoiser_model import restore_image
        import numpy as np

        if args.image:
            if not os.path.isfile(args.image):
                return int(not fail(f"--image not found: {args.image}"))
            img = cv2.imread(args.image)
            if img is None:
                return int(not fail(f"Could not decode {args.image} as an image."))
            label = args.image
        else:
            # deterministic synthetic degraded face-like patch, so this script has
            # zero external dependencies (no test image required to just prove the
            # pipeline runs end to end)
            rng = np.random.default_rng(0)
            img = np.full((256, 256, 3), 150, np.uint8)
            cv2.circle(img, (128, 120), 80, (200, 190, 180), -1)
            cv2.circle(img, (100, 95), 10, (40, 40, 40), -1)
            cv2.circle(img, (156, 95), 10, (40, 40, 40), -1)
            cv2.ellipse(img, (128, 165), (30, 12), 0, 0, 180, (70, 40, 110), -1)
            img = cv2.GaussianBlur(img, (9, 9), 0)
            img = np.clip(img.astype(np.int16) + rng.normal(0, 15, img.shape), 0, 255) \
                .astype(np.uint8)
            label = "generated test image (no --image given)"

        t0 = time.time()
        out = restore_image(net, img, device=device, amp=(device == "cuda"))
        dt = time.time() - t0

        if out.shape != img.shape:
            passed = fail(f"Output shape {out.shape} != input shape {img.shape}")
        elif not np.isfinite(out.astype(np.float32)).all():
            passed = fail("Output contains non-finite values (nan/inf) — model is broken.")
        else:
            changed = float(np.abs(out.astype(np.int16) - img.astype(np.int16)).mean())
            ok(f"Restored {label} in {dt:.2f}s on {device} — "
               f"mean pixel change {changed:.2f}/255")
            if changed < 0.5:
                print(f"        {YELLOW}Note:{RESET} output is nearly identical to input. "
                      f"Fine for a clean input; if this was a blurry/noisy CCTV frame, "
                      f"double-check the checkpoint.")
            out_path = os.path.join(here, "verify_setup_output.png")
            cv2.imwrite(out_path, out)
            print(f"        Wrote {out_path} — open it and look at it.")
    except Exception as e:
        passed = fail(f"{type(e).__name__}: {e}")

    # ---- summary --------------------------------------------------------------
    section("Summary")
    if passed:
        print(f"  {GREEN}{BOLD}All checks passed.{RESET} Safe to start the API: "
              f"python serve_api.py")
        return 0
    else:
        print(f"  {RED}{BOLD}Fix the FAIL items above, then re-run this script.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
