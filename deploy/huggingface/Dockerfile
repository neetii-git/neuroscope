# Hugging Face Space (Docker SDK, free CPU tier). The code comes from GitHub; the model and example patient
# are uploaded to the Space itself under data/ (see DEPLOY_HF.md).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2
WORKDIR /home/user

RUN pip install --user --index-url https://download.pytorch.org/whl/cpu torch
# Bump this number (Settings > Factory rebuild also works) to pull the latest code from GitHub
ARG CODE_VERSION=1
RUN git clone --depth 1 https://github.com/neetii-git/neuroscope.git app
RUN grep -v -E "^(torch|pytest)" app/requirements.txt > /tmp/req.txt && pip install --user -r /tmp/req.txt

COPY --chown=user data /home/user/data
ENV NEUROSCOPE_CKPT=/home/user/data/runs/base/best.pt \
    NEUROSCOPE_RESULTS=/home/user/data/runs/base/results \
    NEUROSCOPE_DEMO=/home/user/data/demo_case

WORKDIR /home/user/app
EXPOSE 7860
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--browser.gatherUsageStats=false"]
