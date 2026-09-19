# Deploying the app

`web/app.py` is a **Streamlit server app**, so it cannot run on GitHub Pages
(which serves only static files). Use a host that runs Python. The easiest is
**Streamlit Community Cloud**, which builds directly from this GitHub repo, free.

---

## Option 0 — GitHub Codespaces (run it *on* GitHub)

Codespaces is a GitHub-hosted dev machine, so the app runs from your repo with no
other service. Good for demos/sharing; it is **not** permanent hosting (it stops
when the codespace stops).

1. Click **Code → Codespaces → Create codespace on main**, or use the badge in the
   README, or open <https://codespaces.new/hslhuang123/stock-agent>.
2. Wait for setup — the dev container installs `requirements.txt` and starts
   Streamlit automatically.
3. The **Ports** tab forwards port **8501**; a preview opens. To share it, set the
   port's visibility to **Public** (Ports → right-click 8501 → Port Visibility).

If the app isn't running, start it in the terminal:

```bash
streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

Codespaces free tier gives personal accounts monthly core-hours; it is not a
24/7 deployment. For that, use Streamlit Community Cloud below.

---

## Option 1 — Streamlit Community Cloud (recommended)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:

   | Field | Value |
   |---|---|
   | Repository | `hslhuang123/stock-agent` |
   | Branch | `main` |
   | Main file path | `web/app.py` |

4. (Optional) **Advanced settings → Python version: 3.11** (matches `runtime.txt`).
5. Click **Deploy**. The first build takes a few minutes.

Streamlit installs `requirements.txt` automatically. **No secrets are needed** —
the app runs fully offline with the *Offline / synthetic* toggle. The optional
LLM analyst needs an OpenAI key: add it under **Manage app → Settings → Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
```

### Things to know

- Free apps **sleep** after inactivity and wake on the next visit.
- Live data comes from Yahoo (yfinance) and can be rate-limited from a datacenter
  IP. If the screen fails, use the **Offline / synthetic** toggle.
- The cloud filesystem is **ephemeral**: the `data/` cache and `reports/` reset
  on restart. That is expected — both are gitignored.

---

## Option 2 — Render / Railway

Both read the committed `Procfile`:

```
web: streamlit run web/app.py --server.port $PORT --server.address 0.0.0.0
```

- **Render:** New → Web Service → connect the repo → Build
  `pip install -r requirements.txt` → Start
  `streamlit run web/app.py --server.port $PORT --server.address 0.0.0.0`.
- **Railway:** New Project → Deploy from GitHub → the `Procfile` is picked up.

---

## Option 3 — Hugging Face Spaces

Create a Space with **SDK: Streamlit**. HF runs `app.py` at the repo root, so add
a one-line shim `app.py`:

```python
import runpy

runpy.run_path("web/app.py", run_name="__main__")
```

---

## Option 4 — Docker (any host)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Why not GitHub Pages?

GitHub Pages serves only static HTML/CSS/JS. This app computes in Python at
request time (screening, backtests, the simulator), so it needs a live Python
process. GitHub is the *source*; Streamlit Cloud (or Render/Railway/HF) is the
*host*.
