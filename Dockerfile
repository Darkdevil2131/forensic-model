# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries the wheels do NOT bundle:
#   libglib2.0-0  -> cv2 (even the headless build links libgthread/libglib)
#   libgl1        -> mediapipe's native _framework_bindings.so
#   libgomp1      -> torch's OpenMP runtime
#   libportaudio2 -> sounddevice, which mediapipe imports
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer is cached when only application code changes.
# mediapipe is installed as a SEPARATE --no-deps step and must not appear in
# requirements.txt -- see the comment at the top of that file for why.
COPY requirements.txt ./
RUN pip install -r requirements.txt \
 && pip install --no-deps mediapipe==0.10.33

# Fail the BUILD rather than the deploy if the cv2/mediapipe/torch combination
# is broken. This is the check that catches a stray opencv-contrib-python.
RUN python -c "import cv2, mediapipe, torch, numpy; print('cv2', cv2.__version__, '| mediapipe', mediapipe.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)"

COPY . .

# serve_api.py defaults to ./denoiser_inference.pth; stated explicitly here so
# the path does not depend on the working directory.
ENV FORENSIC_MODEL=/app/denoiser_inference.pth

# Cosmetic on Railway -- serve_api.py binds 0.0.0.0:$PORT, which Railway injects.
EXPOSE 8000

CMD ["python", "serve_api.py"]
