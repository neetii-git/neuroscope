"""Hugging Face Space entry point (Gradio SDK, free CPU tier).

The Space runs `python app.py`. This script fetches NeuroScope (code, trained model and example patient) from GitHub
and starts the Streamlit app on the port the Space exposes (7860).
"""
import os
import subprocess
import sys

REPO = "https://github.com/neetii-git/neuroscope.git"
HOME = os.path.expanduser("~")
CODE = os.path.join(HOME, "neuroscope")

if not os.path.isdir(os.path.join(CODE, "app")):
    subprocess.run(["git", "clone", "--depth", "1", REPO, CODE], check=True)
else:
    subprocess.run(["git", "-C", CODE, "pull", "--ff-only"], check=False)

data = os.path.join(CODE, "hosting", "data")
env = dict(os.environ,
           NEUROSCOPE_CKPT=os.path.join(data, "runs", "base", "best.pt"),
           NEUROSCOPE_RESULTS=os.path.join(data, "runs", "base", "results"),
           NEUROSCOPE_DEMO=os.path.join(data, "demo_case"),
           OMP_NUM_THREADS="2")
os.chdir(CODE)
os.execvpe(sys.executable, [sys.executable, "-m", "streamlit", "run", "app/app.py",
                            "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true",
                            "--server.enableXsrfProtection=false", "--server.enableCORS=false",
                            "--browser.gatherUsageStats=false"], env)
