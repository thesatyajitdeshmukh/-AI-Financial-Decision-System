# 🚀 AI Financial Decision System — Streamlit Deployment Guide

---

## 📁 Project Structure

After deployment, your project should look like this:

```
ai-financial-system/
│
├── app.py                    ← Main Streamlit app
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Theme & server config
│
├── models/                   ← From your notebook
│   ├── tft_stock_model.ckpt
│   ├── ppo_trading_agent.zip
│   └── finbert/
│       ├── config.json
│       └── pytorch_model.bin
│
└── results/                  ← From your notebook
    └── tft_predictions.csv
```

---

## ✅ STEP 1 — Install Dependencies Locally

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

---

## ✅ STEP 2 — Copy Your Saved Models & Results

From your Google Colab, download the files you saved:
- `models.zip` → extract to `models/` folder
- `results.zip` → extract to `results/` folder

Place these next to `app.py`:

```bash
unzip models.zip    # creates models/ folder
unzip results.zip   # creates results/ folder
```

---

## ✅ STEP 3 — Run Locally to Test

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

You should see the full dashboard with:
- 📊 Market Overview
- ⚙️ Feature Engineering
- 🤖 Trading Signals & Portfolio
- 🎲 Monte Carlo Risk
- 🧠 Model Insights
- 💬 Sentiment Analysis

---

## ✅ STEP 4 — Deploy on Streamlit Community Cloud (Free)

### 4a. Push to GitHub

```bash
# Initialize git repo
git init
git add app.py requirements.txt .streamlit/

# Add your models (if small enough < 100MB each)
git add models/ results/

git commit -m "Initial commit: AI Financial Decision System"

# Create repo on GitHub.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/ai-financial-system.git
git push -u origin main
```

> ⚠️ If `models/` folder is > 100MB, use Git LFS or store models on HuggingFace Hub.

### 4b. Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - Repository: `YOUR_USERNAME/ai-financial-system`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **"Deploy!"**

Your app will be live at:
`https://YOUR_USERNAME-ai-financial-system.streamlit.app`

---

## ✅ STEP 5 — Load Your Saved TFT Model in the App

In `app.py`, add this block to load your real trained model:

```python
import torch
from pytorch_forecasting import TemporalFusionTransformer

@st.cache_resource
def load_tft_model():
    model = TemporalFusionTransformer.load_from_checkpoint("models/tft_stock_model.ckpt")
    model.eval()
    return model

tft = load_tft_model()
```

---

## ✅ STEP 6 — Load Your Real FinBERT Model

Replace the rule-based sentiment with real FinBERT:

```python
from transformers import pipeline

@st.cache_resource
def load_finbert():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device=-1  # use 0 for GPU
    )

sentiment_model = load_finbert()

# Usage
results = sentiment_model(headlines)
```

---

## ✅ STEP 7 — Load Your PPO RL Agent

```python
from stable_baselines3 import PPO

@st.cache_resource
def load_ppo():
    return PPO.load("models/ppo_trading_agent.zip")

ppo_model = load_ppo()

# Run agent on new data
obs = env.reset()
action, _ = ppo_model.predict(obs)
```

---

## ⚙️ Environment Variables (Secrets)

For API keys or tokens, use Streamlit Secrets:

**Locally** — create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_huggingface_token"
```

**On Streamlit Cloud** — go to App Settings → Secrets and paste:
```toml
HF_TOKEN = "your_huggingface_token"
```

Access in code:
```python
import streamlit as st
token = st.secrets["HF_TOKEN"]
```

---

## 🐳 Optional — Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ai-financial-app .
docker run -p 8501:8501 ai-financial-app
```

---

## 🌐 Other Deployment Options

| Platform | Command | URL |
|----------|---------|-----|
| **Streamlit Cloud** | Push to GitHub + connect | share.streamlit.io (Free) |
| **Heroku** | `heroku create` + git push | your-app.herokuapp.com |
| **Railway** | Connect GitHub repo | railway.app |
| **Hugging Face Spaces** | Create Space (Streamlit SDK) | huggingface.co/spaces |
| **Google Cloud Run** | `gcloud run deploy` | Custom domain |
| **AWS EC2** | `streamlit run app.py` | Your IP:8501 |

---

## 🐛 Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: ta` | `pip install ta` |
| `yfinance` timeout | Add retry: `yf.download(..., timeout=60)` |
| TFT checkpoint not found | Check model path is correct |
| Memory error on Cloud | Use `@st.cache_data` for heavy data |
| Port already in use | `streamlit run app.py --server.port 8502` |
| FinBERT too slow | Use `@st.cache_resource` to load once |

---

## 📦 Packages Summary

| Package | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `yfinance` | Live stock data |
| `ta` | Technical indicators |
| `pytorch-forecasting` | TFT model |
| `stable-baselines3` | PPO RL agent |
| `transformers` | FinBERT sentiment |
| `torch-geometric` | GNN model |
| `scikit-learn` | GMM, preprocessing |
| `matplotlib` | Charts & plots |

---

*Generated for: AI Financial Decision System — TFT + FinBERT + PPO + GNN + Monte Carlo*
