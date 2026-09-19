FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libegl1 \
        libgles2 \
        libglib2.0-0 \
        libgomp1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

# MediaPipe also declares the GUI-enabled OpenCV wheel. Reinstall the headless
# build last so server containers do not depend on OpenCV's GUI/X11 stack.
RUN python -m pip install --force-reinstall --no-deps \
    opencv-python-headless==4.14.0.94

RUN addgroup --system handrom \
    && adduser --system --ingroup handrom handrom

COPY --chown=handrom:handrom . .

USER handrom

# Exercise the same native-library and model-loading path used by Analyze images.
# A missing Linux shared library must fail the image build, not a patient session.
RUN python -c "from handrom.hand_detector import create_hand_landmarker; landmarker = create_hand_landmarker('models/hand_landmarker.task'); landmarker.close()"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
