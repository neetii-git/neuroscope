# Free hosting on Streamlit Community Cloud

Free, no card, sign in with GitHub. Link: `https://<chosen-name>.streamlit.app`.
Up to 2.7 GB memory (NeuroScope peaks at about 1.9 GB). Sleeps after 12 hours without visits; a visitor wakes it with one click.

1. share.streamlit.io > **Continue with GitHub** and authorise Streamlit.
2. **Create app** > **Deploy a public app from GitHub**.
3. Repository `neetii-git/neuroscope`, branch `main`, main file path `app/app.py`.
4. App URL: choose the name, e.g. `neuroscope`.
5. Advanced settings: Python 3.11. **Deploy**.

The model and example patient are read from `hosting/data` in the repository. Every push to `main` redeploys automatically.
First build takes about 5 to 10 minutes (PyTorch install).
