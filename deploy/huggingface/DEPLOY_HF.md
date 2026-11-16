# Free hosting on Hugging Face Spaces

Result: NeuroScope at `https://<your-username>-neuroscope.hf.space`, free, no card.
On the free tier the Space sleeps after about 48 hours without visitors; the next visit wakes it in about a minute.

## 1. Account
Sign up at huggingface.co (free). Verify your email.

## 2. Create the Space
1. huggingface.co > your profile picture > **New Space**.
2. Space name: `neuroscope`. License: MIT. **Space SDK: Docker**, template **Blank**.
3. Hardware: **CPU basic (free)**. Visibility: **Public**.
4. **Create Space**.

## 3. Add the two setup files
In the Space, open **Files** > **+ Add file** > **Create a new file**:
1. File name `Dockerfile`, paste the contents of `deploy/huggingface/Dockerfile` from GitHub, **Commit**.
2. Open the existing `README.md` > **Edit**, replace everything with `deploy/huggingface/README.md`, **Commit**.

## 4. Model and example patient
Nothing to upload: they are in this repository under `hosting/data` and the Space downloads them when it builds.

## 5. Wait for the build
The Space builds automatically (**Logs** tab shows progress), about 5 to 10 minutes. When it says **Running**, open
`https://<your-username>-neuroscope.hf.space`. That link shows only the app, full screen.

The first "Load example patient" takes 1 to 2 minutes; after that it is cached until the Space restarts.

## Updating after code changes
Code is pulled from GitHub when the Space builds. To get new code: Space **Settings** > **Factory rebuild**.

## Notes
- The example patient comes from the BraTS 2020 dataset. Putting it in a public Space redistributes it; check the dataset's terms
  (or leave `demo_case` out, and visitors use uploads only).
- Uploaded scans are processed in memory for the visitor's session and are not stored.
