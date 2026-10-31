# MetaRisk XAI — Next.js + FastAPI Setup Guide

## Project Structure After Changes

```
Explainable-AI-XAI-for-Metabolic-Syndrome-Risk/
│
├── server.py              ← Updated FastAPI backend (REST API)
├── main.py                ← Training pipeline (unchanged)
├── validatiion.py         ← Fixed validation pipeline
├── requirements.txt       ← Add: fastapi uvicorn[standard]
├── railway.json           ← Railway deployment config (backend)
│
└── frontend/              ← NEW: Next.js app
    ├── package.json
    ├── next.config.mjs
    ├── tailwind.config.ts
    ├── postcss.config.js
    ├── tsconfig.json
    ├── vercel.json
    ├── .env.local         ← API URL config (don't commit)
    ├── app/
    │   ├── globals.css
    │   ├── layout.tsx
    │   ├── page.tsx           (landing page)
    │   ├── patient/page.tsx   (patient portal)
    │   └── clinician/page.tsx (clinician dashboard)
    ├── components/
    │   ├── Navbar.tsx
    │   ├── RiskGauge.tsx
    │   ├── ShapChart.tsx
    │   ├── SurvivalCurve.tsx
    │   ├── RecourseCard.tsx
    │   ├── FairnessChart.tsx
    │   ├── BenchmarkTable.tsx
    │   └── BatchUpload.tsx
    └── lib/
        ├── api.ts
        └── utils.ts
```

---

## Step 1: Update Python requirements

Add to your `requirements.txt`:
```
fastapi
uvicorn[standard]
python-multipart
```

---

## Step 2: Train models (if not done)

```bash
python main.py
```

This generates model.pkl, model_rf.pkl, model_xgb.pkl, model_lgb.pkl.

---

## Step 3: Start the FastAPI backend

```bash
# In the project root:
uvicorn server:app --reload --port 8000
```

Test it: open http://localhost:8000/health
You should see: `{"status":"ok","models_loaded":["rf","xgb","lgb","best"]}`

---

## Step 4: Set up the Next.js frontend

```bash
# In a NEW terminal:
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## Step 5: Deploy to Production

### Backend → Railway (free tier)

1. Go to https://railway.app → New Project → Deploy from GitHub
2. Select your repo → set Root Directory to `/` (the Python root)
3. Add environment variables:
   - `FRONTEND_URL` = your Vercel URL (e.g. https://metarisk.vercel.app)
4. Railway auto-detects railway.json and runs:
   - Build: `pip install -r requirements.txt && python main.py`
   - Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Copy your Railway URL (e.g. https://metarisk-xai.railway.app)

### Frontend → Vercel (free)

1. Go to https://vercel.com → New Project → Import from GitHub
2. Set Root Directory to `frontend`
3. Add Environment Variable:
   - `NEXT_PUBLIC_API_URL` = your Railway URL (e.g. https://metarisk-xai.railway.app)
4. Deploy → done!

---

## CORS Note

In server.py, the CORS middleware already allows `*.vercel.app`.
If you use a custom domain, add it to the `allow_origins` list.

---

## .gitignore additions

Add to your existing .gitignore:
```
# Frontend
frontend/node_modules/
frontend/.next/
frontend/.env.local
```
