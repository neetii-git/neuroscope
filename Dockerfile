# NeuroScope web app (CPU). Model files and the example patient are mounted at run time, not baked into the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2
WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA libraries)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN grep -v -E "^(torch|pytest)" requirements.txt > req.txt && pip install -r req.txt

COPY neuroscope ./neuroscope
COPY app ./app
COPY configs ./configs
COPY .streamlit ./.streamlit

ENV NEUROSCOPE_CKPT=/data/runs/base/best.pt \
    NEUROSCOPE_RESULTS=/data/runs/base/results \
    NEUROSCOPE_DEMO=/data/demo_case

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
