# Setup: run, publish and back up (about 20 minutes)

You need **both**:
- a **public GitHub repository**, which is your backup and what you submit;
- a **public app link** that opens without signing in.

The activity website has the same steps with troubleshooting: <https://snsie.github.io/classification-activity/hosting.html>

| | Route A: GitHub Pages (recommended) | Route B: Streamlit Community Cloud |
|---|---|---|
| You write | HTML + JavaScript (`app/index.html`, `app/app.js`) | Python (`streamlit_app.py`) |
| Good if you… | like web pages, or want full design freedom | used Streamlit before |
| Runs Python online? | No: static files only. The app reads `results/` | Yes |

Your Python analysis runs on your computer either way.

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
   You need git on your computer: the workflow checks that you committed your decisions before you ran each stage.

## 3. Run it on your computer
- **Route A:** `npm run serve` (or `python -m http.server 8000`), then open <http://localhost:8000/app/>. Opening the HTML file directly won't load the results.
- **Route B:** `streamlit run streamlit_app.py`

## 4. Publish the untouched starter now
- **Route A:** on GitHub, open **Settings → Pages**, set **Source: Deploy from a branch** with branch `main` and folder `/ (root)`, and click **Save**. After 1–2 minutes your link is `https://<username>.github.io/<repo>/`.
- **Route B:** at [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick your repository, branch `main` and file `streamlit_app.py`, then click **Deploy**.

## 5. Check it like a reviewer
Open your link in a **private/incognito window**. It must load without signing in. Put the link on the **first line** of `README.md`, then commit and push.

## Push updates
```bash
git add .
git commit -m "Section 2 works"
git push
```
Both hosts update a minute or two after you push. Commit `results/` too: the app reads it from GitHub.

## Something broke?
- **The Pages link shows a 404:** wait 2 minutes, and check that Pages is set to `main` and `/ (root)`.
- **The page is blank or says no results:** check that `results/` is committed and pushed, and that your code loads results through `m11.load`.
- **Streamlit says "ModuleNotFoundError":** add the package to `requirements.txt` and push.
- **The Streamlit app is asleep:** open the link and click the button to wake it; it sleeps after a few days without visitors.
- **`import lightgbm` fails on a Mac with a message about `libomp`:** run `brew install libomp`, then try again.
- **Still stuck:** switch to the other route. Both read the same `results/` files.
