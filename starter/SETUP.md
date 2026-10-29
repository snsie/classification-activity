# Setup: run, publish and back up

You need **both**:
- a **public GitHub repository**, which is your backup and what you submit;
- a **public app link** that opens without signing in.

## Pick your app: Streamlit or Node.js
The starter has the same four plain sections twice. Build and host **one** of them. If you like, delete the other app (`streamlit_app.py`, or `web/`, `index.html`, `.nojekyll` and `package.json`), but keep `m11.py`.

| | Streamlit (Python) | Node.js (JavaScript) |
|---|---|---|
| Your app | `streamlit_app.py` | `web/app.js` (page: `web/index.html`, `web/style.css`) |
| Metric helpers | `m11.py` | `web/m11.js` (the same functions, ported) |
| Run it on your computer | `streamlit run streamlit_app.py` | `npm start`, then open <http://localhost:8766/> |
| Free hosting | Streamlit Community Cloud | GitHub Pages |

Both routes earn the same credit.

The activity website has the same steps with troubleshooting: <https://snsie.github.io/classification-activity/hosting.html>

Either way, your Python analysis runs on your computer, and the hosted app only reads the `results/` files it exported.

## 1. Install
You need Python 3.11 or newer and git, plus Node.js 20 or newer if you choose the Node app.
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
streamlit run streamlit_app.py       # Streamlit
npm start                            # Node.js (Node 20 or newer), then open http://localhost:8766/
```

## 4. Publish the untouched starter now
- **Streamlit:** at [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick your repository, branch `main` and file `streamlit_app.py`, then click **Deploy**.
- **Node.js:** on your repository's GitHub page, open **Settings → Pages**. Under **Build and deployment**, set Source **Deploy from a branch**, Branch `main` and folder `/ (root)`, then click **Save**. After a minute or two your link appears at the top of that page: `https://<username>.github.io/<repo>/`. GitHub Pages serves your files as they are: `npm start` is only for your computer.

## 5. Check it like a reviewer
Open your link in a **private/incognito window**. It must load without signing in. Put the link on the **first line** of `README.md`, then commit and push.

## Push updates
```bash
git add .
git commit -m "Section 2 works"
git push
```
Your host updates a minute or two after you push. Commit `results/` too: the app reads it from GitHub.

## Something broke?
- **The app says no results:** check that `results/` is committed and pushed, and that your code loads results through `m11.load`.
- **Streamlit says "ModuleNotFoundError":** add the package to `requirements.txt` and push.
- **The Streamlit app is asleep:** open the link and click the button to wake it; it sleeps after a few days without visitors.
- **The GitHub Pages link shows a 404:** check that the repository is public and that Settings → Pages uses branch `main` and folder `/ (root)`. The first publish can take a few minutes.
- **The GitHub Pages app shows old numbers:** Pages updates a minute or two after you push, and your browser may cache the old files; reload with Ctrl+Shift+R (Mac: Cmd+Shift+R).
- **The Node app shows a blank page:** open the browser console (F12) and read the first error; it names the file and line in `web/`. GitHub Pages has no npm: load any library you add as a browser module saved in `web/` (or from a CDN), not with `npm install`.
- **`import lightgbm` fails on a Mac with a message about `libomp`:** run `brew install libomp`, then try again.
