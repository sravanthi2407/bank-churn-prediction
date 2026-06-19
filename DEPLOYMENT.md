# 🚀 Deployment Guide — Bank Customer Churn Predictor

## Prerequisites
- Python 3.9+
- pip package manager
- Git

---

## Option 1: Streamlit Cloud (Recommended — Free)

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Bank Churn ML Project"
git remote add origin https://github.com/yourusername/bank-churn-prediction.git
git push -u origin main
```

2. **Deploy on share.streamlit.io**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub repository
   - Set Main file path: `app.py`
   - Click Deploy

---

## Option 2: Render

Create `render.yaml`:
```yaml
services:
  - type: web
    name: bank-churn-predictor
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt && python preprocess.py && python train_model.py
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

---

## Option 3: Hugging Face Spaces

Add this YAML header to README.md:
```yaml
---
title: Bank Customer Churn Predictor
emoji: 🏦
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---
```

Then push all files to your HF Space repository.

---

## Option 4: Local Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python preprocess.py && python train_model.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t bank-churn .
docker run -p 8501:8501 bank-churn
```
