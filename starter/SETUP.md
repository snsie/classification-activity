# Setup: run, publish and back up

You need **both**:
- a **public GitHub repository**, which is your backup and what you submit;
- a **public app link** on Streamlit Community Cloud that opens without signing in.

The activity website has the same steps with troubleshooting: <https://snsie.github.io/classification-activity/hosting.html>

Your Python analysis runs on your computer. The hosted app only reads the `results/` files it exported.

## 1. Install
You need Python 3.11 or newer and git.
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check.py --stage setup
```

## 2. Make your repository
1. Unzip this starter. Its folder is your project.
2. On GitHub, click **New repository**, give it a name (e.g. `judge-the-model`) and make it **public**. Don't add a README; the starter has one.
3. In the starter folder, run:
   ```bash
   git init -b main
   git add .
   git commit -m "Starter"
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```

## 3. Run it on your computer
```bash
streamlit run streamlit_app.py
```

## 4. Publish the untouched starter now
At [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick your repository, branch `main` and file `streamlit_app.py`, then click **Deploy**.

## 5. Check it like a reviewer
Open your link in a **private/incognito window**. It must load without signing in. Put the link on the **first line** of `README.md`, then commit and push.

## Push updates
```bash
git add .
git commit -m "Section 2 works"
git push
```
Streamlit updates a minute or two after you push. Commit `results/` too: the app reads it from GitHub.

## Something broke?
- **The app says no results:** check that `results/` is committed and pushed, and that your code loads results through `m11.load`.
- **Streamlit says "ModuleNotFoundError":** add the package to `requirements.txt` and push.
- **The Streamlit app is asleep:** open the link and click the button to wake it; it sleeps after a few days without visitors.
