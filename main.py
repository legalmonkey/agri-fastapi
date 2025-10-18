from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import math
import numpy as np
import pandas as pd
import joblib
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# ------------------------------ Project paths ------------------------------
PROJ_ROOT = os.getcwd()
ART_DIR = os.path.join(PROJ_ROOT, "artifacts_yield")
MODEL_DIR = os.path.join(ART_DIR, "lstm_seq_model")
META_PATH = os.path.join(ART_DIR, "lstm_seq_meta.pkl")

RULES_PATH = os.path.join(PROJ_ROOT, "rules", "crop_reco.json")
PROC_DIR = os.path.join(PROJ_ROOT, "processed_training_csvs")

# ------------------------------ Globals ------------------------------
_ready = {"ok": False, "reason": "initializing"}
_lstm_model = None
_seq_meta: Dict = {}
_RULES_CACHE: Dict = {}
_ENRICHED_DF: Optional[pd.DataFrame] = None

# ------------------------------ Utilities ------------------------------
def _norm_text(s: str) -> str:
    return str(s).strip().lower().replace("&", "and").replace(".", "").replace("-", " ") if s is not None else ""

def _latest_enriched_path() -> Optional[str]:
    if not os.path.isdir(PROC_DIR):
        return None
    runs = [os.path.join(PROC_DIR, d) for d in os.listdir(PROC_DIR) if d.startswith("run_")]
    if not runs:
        return None
    runs.sort()
    cand = os.path.join(runs[-1], "01_enriched_base.csv")
    return cand if os.path.isfile(cand) else None

def _standardize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    canon = ["Rainfall_sum","Tavg_mean","Tmax_mean","Tmin_mean","ET0_sum","GDD_sum"]
    renames = {}
    for v in canon:
        choices = []
        if v in df.columns: choices.append(v)
        if f"{v}_x" in df.columns: choices.append(f"{v}_x")
        if f"{v}_y" in df.columns: choices.append(f"{v}_y")
        if not choices: continue
        nn = {c: df[c].notna().sum() for c in choices}
        best = sorted(choices, key=lambda c: (-nn[c], 0 if c==v else (1 if c.endswith('_x') else 2)))[0]
        if best != v: renames[best] = v
    if renames:
        df = df.rename(columns=renames)
    return df

def _load_enriched_base() -> pd.DataFrame:
    global _ENRICHED_DF
    if _ENRICHED_DF is not None:
        return _ENRICHED_DF
    path = _latest_enriched_path()
    if path and os.path.isfile(path):
        df = pd.read_csv(path)
        df = _standardize_feature_names(df)
        if "statenorm" not in df.columns and "statename" in df.columns:
            df["statenorm"] = df["statename"].astype(str).map(_norm_text)
        if "districtnorm" not in df.columns and "districtname" in df.columns:
            df["districtnorm"] = df["districtname"].astype(str).map(_norm_text)
        _ENRICHED_DF = df
        return df
    _ENRICHED_DF = pd.DataFrame()
    return _ENRICHED_DF

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))
    a = math.sin(dlat/2)**2 + math.cos(math.radians(float(lat1))) * math.cos(math.radians(float(lat2))) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ------------------------------ Rules loader ------------------------------
def load_crop_rules() -> dict:
    global _RULES_CACHE
    if _RULES_CACHE:
        return _RULES_CACHE
    if not os.path.isfile(RULES_PATH):
        _RULES_CACHE = {
            "rice": {
                "fertilizer_blend": {"npk": "NPK 18-46-0 + 0-0-60 (split)", "note": "Adjust by soil test; split N and K."},
                "irrigation": {"title": "Sprinkler System", "subtitle": "Optimized for cereals/loams"},
                "pesticides": ["Imidacloprid", "Fipronil", "Chlorantraniliprole"]
            }
        }
        return _RULES_CACHE
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        _RULES_CACHE = json.load(f)
    return _RULES_CACHE

def recommend_for_crop(crop: str) -> dict:
    rules = load_crop_rules()
    rule = rules.get(_norm_text(crop))
    if not rule:
        rule = {
            "fertilizer_blend": {"npk": "NPK 18-46-0 (DAP) + Urea split", "note": "Refine with soil test maps"},
            "irrigation": {"title": "Sprinkler System", "subtitle": "Uniform coverage"},
            "pesticides": ["Imidacloprid", "Mancozeb", "Chlorantraniliprole"]
        }
    return rule

# ------------------------------ Georesolver ------------------------------
def resolve_lat_lon(state: str, district: str):
    df = _load_enriched_base()
    if df.empty:
        return (22.9734, 78.6569)
    st, dt = _norm_text(state), _norm_text(district)
    cand = df[(df.get("statenorm","")==st) & (df.get("districtnorm","")==dt)]
    if "lat" in df.columns and "lon" in df.columns and not cand.empty:
        lat = pd.to_numeric(cand["lat"], errors="coerce").dropna()
        lon = pd.to_numeric(cand["lon"], errors="coerce").dropna()
        if not lat.empty and not lon.empty:
            return float(lat.iloc[0]), float(lon.iloc[0])
    st_rows = df[df.get("statenorm","")==st]
    if not st_rows.empty and {"lat","lon"}.issubset(st_rows.columns):
        lat = pd.to_numeric(st_rows["lat"], errors="coerce").dropna()
        lon = pd.to_numeric(st_rows["lon"], errors="coerce").dropna()
        if not lat.empty and not lon.empty:
            return float(lat.iloc[0]), float(lon.iloc[0])
    return (22.9734, 78.6569)

# ------------------------------ NASA POWER ------------------------------
def _clip_temp_celsius(vals: List[float]) -> List[float]:
    out = []
    for v in vals:
        if v is None:
            continue
        try:
            f = float(v)
        except:
            continue
        if f < -90 or f > 70:
            continue
        out.append(f)
    return out

def fetch_prev_week_weather(lat: float, lon: float, end_date: Optional[str]) -> Dict[str, float]:
    if end_date:
        try:
            ref = datetime.strptime(end_date, "%Y-%m-%d")
        except Exception:
            ref = datetime.utcnow()
    else:
        ref = datetime.utcnow()
    end_str = ref.strftime("%Y%m%d")
    start_str = (ref - timedelta(days=8)).strftime("%Y%m%d")

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start": start_str,
        "end": end_str,
        "parameters": "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN",
        "community": "ag",
        "temporal": "daily",
        "format": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        param = r.json().get("properties", {}).get("parameter", {})
        pre, tavg, tmax, tmin = param.get("PRECTOTCORR", {}), param.get("T2M", {}), param.get("T2M_MAX", {}), param.get("T2M_MIN", {})

        def arr(d):
            if not isinstance(d, dict):
                return []
            vals = []
            for v in d.values():
                try:
                    vals.append(float(v))
                except:
                    pass
            return vals

        rain = [v for v in arr(pre) if v is not None and v > -0.001]
        rain_sum = round(sum([max(0.0, v) for v in rain]), 3)

        tavg_vals = _clip_temp_celsius(arr(tavg))
        tmax_vals = _clip_temp_celsius(arr(tmax))
        tmin_vals = _clip_temp_celsius(arr(tmin))

        tavg_mean = round(sum(tavg_vals)/len(tavg_vals), 3) if tavg_vals else 0.0
        tmax_mean = round(sum(tmax_vals)/len(tmax_vals), 3) if tmax_vals else 0.0
        tmin_mean = round(sum(tmin_vals)/len(tmin_vals), 3) if tmin_vals else 0.0

        return {
            "Rainfall_sum": rain_sum,
            "Tavg_mean": tavg_mean,
            "Tmax_mean": tmax_mean,
            "Tmin_mean": tmin_mean,
            "ET0_sum": 0.0,
            "GDD_sum": 0.0
        }
    except Exception:
        return {"Rainfall_sum": 0.0, "Tavg_mean": 0.0, "Tmax_mean": 0.0, "Tmin_mean": 0.0, "ET0_sum": 0.0, "GDD_sum": 0.0}

# ------------------------------ Historical sequence ------------------------------
def _nearest_bucket_frame(lat: float, lon: float, crop: str) -> (List[str], np.ndarray):
    df = _load_enriched_base()
    if df.empty:
        return ["Rainfall_sum","Tavg_mean","Tmax_mean","Tmin_mean","ET0_sum","GDD_sum"], np.zeros((0,6), dtype=np.float32)

    sub = df[df["crop"].astype(str).str.lower()==_norm_text(crop)].copy()
    if sub.empty:
        sub = df.copy()

    feats = [c for c in ["Rainfall_sum","Tavg_mean","Tmax_mean","Tmin_mean","ET0_sum","GDD_sum"] if c in sub.columns]
    for c in feats:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    if {"lat","lon"}.issubset(sub.columns):
        sub = sub.dropna(subset=["lat","lon"]).copy()
        sub["km"] = sub.apply(lambda r: _haversine(lat, lon, r["lat"], r["lon"]), axis=1)
        sub = sub.sort_values("km")

    if "cropyear" in sub.columns:
        sub = sub.sort_values(["cropyear"])

    X_seq = sub[feats].values.astype(np.float32)
    return feats, X_seq

# ------------------------------ Model load ------------------------------
def _safe_load_model():
    global _lstm_model, _seq_meta, _ready
    try:
        _lstm_model = load_model(MODEL_DIR)
        _seq_meta = joblib.load(META_PATH)
        _ready["ok"] = True
        _ready["reason"] = "ready"
    except Exception as e:
        _ready["ok"] = False
        _ready["reason"] = f"startup failed: {e}"

_safe_load_model()

# ------------------------------ Core prediction (ML math unchanged) ------------------------------
def _predict_core(state: str, district: str, crop: str, land_area: float, end_date: Optional[str]):
    if not _ready.get("ok", False):
        raise RuntimeError(f"Model not ready: {_ready.get('reason')}")

    lat, lon = resolve_lat_lon(state, district)
    feats_hist, X_seq = _nearest_bucket_frame(lat, lon, crop)

    week = fetch_prev_week_weather(lat, lon, end_date=end_date)
    wk_vec = [float(week.get(f, 0.0)) for f in _seq_meta["seq_features"]]

    X_seq = np.vstack([X_seq, np.array(wk_vec, dtype=np.float32)])

    cap_len = int(_seq_meta["cap_len"]); max_len = int(_seq_meta["max_len"])
    cap = min(cap_len, max_len)
    X_cap = X_seq[-cap:] if len(X_seq) > cap else X_seq
    X_padded = pad_sequences([X_cap], maxlen=max_len, dtype="float32", padding="pre", truncating="pre")
    X_padded = np.nan_to_num(X_padded, nan=0.0, posinf=0.0, neginf=0.0)

    yhat = _lstm_model.predict(X_padded, verbose=0)
    if yhat.ndim == 3:
        yhat_seq = yhat[:, :, 0][0]
    elif yhat.ndim == 2:
        yhat_seq = yhat[0, :]
    else:
        yhat_seq = np.ravel(yhat)

    valid_mask = (X_padded[0].sum(axis=1) != 0)
    idxs = np.where(valid_mask)[0]
    if idxs.size > 0:
        y_pred_unit = float(yhat_seq[idxs[-1]])
    else:
        nz = yhat_seq[np.nonzero(yhat_seq)]
        y_pred_unit = float(nz.mean()) if nz.size else float(yhat_seq[-1])

    # Interpret model per-area result; UI shows tonnes/acre
    yield_per_area_pred = y_pred_unit
    production_pred = yield_per_area_pred * float(land_area)

    week_display = {
        "Rainfall_sum (in mm)": float(week.get("Rainfall_sum", 0.0)),
        "Average Mean Temp": float(week.get("Tavg_mean", 0.0)),
        "Mean Maximum Temp": float(week.get("Tmax_mean", 0.0)),
        "Mean Minimum Temp": float(week.get("Tmin_mean", 0.0)),
    }

    return {
        "state": state, "district": district, "crop": crop,
        "lat": lat, "lon": lon,
        "yield_per_area_pred": yield_per_area_pred,  # displayed as tonnes/acre
        "area_input": land_area,                      # acres
        "production_pred": production_pred,           # tonnes
        "week_features": week_display,
        "end_date_used": end_date
    }

# ------------------------------ UI: Full HTML form ------------------------------
@app.get("/", response_class=HTMLResponse)
def form():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agricultural Analysis</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">

  <style>
    :root{
      --bg:#070b0a; --card: rgba(15, 25, 20, 0.35); --card-border: rgba(255,255,255,0.08);
      --accent:#30d158; --text:#e6f5ea; --muted:#a9b8ae;
      --input-bg: rgba(255,255,255,0.06); --input-border: rgba(255,255,255,0.12);
      --input-focus: rgba(48, 209, 88, 0.55);
    }
    *{ box-sizing:border-box; }
    html,body{
      height:100%; margin:0;
      font-family:"Inter",system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background: radial-gradient(1800px 1000px at 60% 0%, #0f1a14 0%, #0b130f 40%, var(--bg) 80%) no-repeat, var(--bg);
      color:var(--text);
    }
    .container{ min-height: 100%; width: 100%; display: flex; align-items: flex-start; justify-content: center; padding: 48px 24px; }
    .panel{
      width: 100%; max-width: 1200px; padding: 46px 36px 40px; border-radius: 18px; background: var(--card);
      border: 1px solid var(--card-border); backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
      box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06); position: relative;
    }
    .home-btn{
      position: absolute; top: 18px; right: 18px; display:inline-flex; align-items:center; gap:10px; padding:10px 14px; border-radius:12px; color: var(--text);
      text-decoration:none; background: rgba(48, 209, 88, 0.12); border:1px solid rgba(48, 209, 88, 0.22); transition: all .2s ease;
    }
    h1{ margin:6px 0 8px; font-size: clamp(32px, 3.6vw, 56px); text-align: center; font-weight: 800; }
    .title-plain { color: #ecfff3; } .title-accent { color: var(--accent); }
    .subtitle{ margin: 0 0 30px; text-align:center; color: var(--muted); font-size: 16px; }

    form{ display:grid; gap: 18px; margin-top: 8px; }
    .field{ display:flex; flex-direction:column; gap:10px; }
    .label{ display:flex; align-items:center; gap:10px; font-weight:600; color:#d6edde; letter-spacing:.2px; }
    .label small{ color:#9eb1a6; font-weight:500; }

    .input{
      width:100%; padding:16px 16px; border-radius:12px; border:1px solid var(--input-border); background: var(--input-bg);
      color: var(--text); outline:none; transition: border-color .2s, box-shadow .2s, background .2s; font-size:16px;
    }
    .input::placeholder{ color:#94a89c; }
    .input:focus{ border-color:var(--input-focus); box-shadow:0 0 0 4px rgba(48,209,88,0.15); background:rgba(255,255,255,0.09); }

    .row{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }

    .btn{
      margin-top: 8px; padding: 18px 20px; width:100%; border:none; border-radius:14px; font-size:18px; font-weight:700; color:#052d14;
      background: linear-gradient(90deg, #28d17a 0%, #30d158 45%, #28d17a 100%); cursor:pointer; transition: transform .08s, filter .2s, box-shadow .2s;
      box-shadow: 0 10px 24px rgba(48,209,88,0.25), inset 0 1px 0 rgba(255,255,255,0.35);
    }
    .btn:hover{ filter: brightness(1.03); }
    .btn:active{ transform: translateY(1px); }

    /* Safe, accessible select styling */
    select.input{
      color:#e6f5ea; background-color: rgba(255,255,255,0.06);
      max-width: 100%;
      border-color: var(--input-border);
    }
    select.input option{
      color:#0b130f; background:#ffffff; /* always visible option text */
    }
  </style>
</head>
<body>
  <div class="container">
    <section class="panel" aria-label="Agricultural Analysis Form">
      <a class="home-btn" href="#"><span aria-hidden="true">🏠</span><span>Home</span></a>

      <h1><span class="title-plain">Agricultural</span><span class="title-accent"> Analysis</span></h1>
      <p class="subtitle">Enter farm details to receive AI‑powered insights</p>

      <form id="farmForm" method="post" action="/predict">
        <div class="row">
          <div class="field">
            <label class="label" for="state"><span class="dot"></span> State</label>
            <input class="input" id="state" name="state" type="text" placeholder="e.g., Punjab" required />
          </div>
          <div class="field">
            <label class="label" for="district"><span class="dot"></span> District</label>
            <input class="input" id="district" name="district" type="text" placeholder="e.g., Ludhiana" required />
          </div>
        </div>

        <div class="field">
          <label class="label" for="crop"><span class="dot"></span> Crop Type</label>
          <select class="input" id="crop" name="crop" required>
            <option value="" disabled selected>Select a crop</option>
            <option>Arecanut</option>
            <option>Arhar/Tur</option>
            <option>Bajra</option>
            <option>Banana</option>
            <option>Barley</option>
            <option>Black pepper</option>
            <option>Cardamom</option>
            <option>Cashewnut</option>
            <option>Castor seed</option>
            <option>Coconut</option>
            <option>Coriander</option>
            <option>Cotton(lint)</option>
            <option>Cowpea(Lobia)</option>
            <option>Dry chillies</option>
            <option>Dry ginger</option>
            <option>Garlic</option>
            <option>Ginger</option>
            <option>Gram</option>
            <option>Groundnut</option>
            <option>Guar seed</option>
            <option>Horse-gram</option>
            <option>Jowar</option>
            <option>Jute</option>
            <option>Khesari</option>
            <option>Linseed</option>
            <option>Maize</option>
            <option>Mango</option>
            <option>Masoor</option>
            <option>Mesta</option>
            <option>Moong(Green Gram)</option>
            <option>Moth</option>
            <option>Niger seed</option>
            <option>Oilseeds total</option>
            <option>Onion</option>
            <option>Other  Rabi pulses</option>
            <option>Other Cereals & Millets</option>
            <option>Other Kharif pulses</option>
            <option>Peas & beans (Pulses)</option>
            <option>Potato</option>
            <option>Ragi</option>
            <option>Rapeseed &Mustard</option>
            <option>Rice</option>
            <option>Safflower</option>
            <option>Sannhamp</option>
            <option>Sesamum</option>
            <option>Small millets</option>
            <option>Soyabean</option>
            <option>Sugarcane</option>
            <option>Sunflower</option>
            <option>Sweet potato</option>
            <option>Tapioca</option>
            <option>Tobacco</option>
            <option>Turmeric</option>
            <option>Urad</option>
            <option>Wheat</option>
            <option>other oilseeds</option>
          </select>
        </div>

        <div class="row">
          <div class="field">
            <label class="label" for="land"><span class="dot"></span> Land Size (Acres)</label>
            <input class="input" id="land" name="land_area" type="number" inputmode="decimal" step="0.01" min="0" placeholder="Enter land size in acres" required />
          </div>
          <div class="field">
            <label class="label" for="endDate"><span class="dot"></span> End Date <small>(for real-time weather data purposes)</small></label>
            <input class="input" id="endDate" name="end_date" type="text" inputmode="numeric" placeholder="YYYY-MM-DD" pattern="\\d{4}-\\d{2}-\\d{2}" title="Enter date as YYYY-MM-DD (e.g., 2025-09-21)" />
          </div>
        </div>

        <button class="btn" type="submit">Run AI Analysis</button>
      </form>
    </section>
  </div>
</body>
</html>
    """

@app.get("/health", response_class=HTMLResponse)
def health():
    return "ok"

@app.get("/debug", response_class=HTMLResponse)
def debug():
    global _ready, _lstm_model, _seq_meta
    model_status = "Model loaded" if _lstm_model is not None else "Model not loaded"
    meta_status = f"Metadata loaded: {len(_seq_meta)} keys" if _seq_meta else "Metadata not loaded"
    
    return f"""
    <html>
    <head><title>Debug Info</title></head>
    <body>
    <h2>Debug Information</h2>
    <p><strong>Ready Status:</strong> {_ready}</p>
    <p><strong>Model Status:</strong> {model_status}</p>
    <p><strong>Metadata Status:</strong> {meta_status}</p>
    <p><strong>Model Directory:</strong> {MODEL_DIR}</p>
    <p><strong>Meta Path:</strong> {META_PATH}</p>
    <p><strong>Model Dir Exists:</strong> {os.path.exists(MODEL_DIR)}</p>
    <p><strong>Meta File Exists:</strong> {os.path.exists(META_PATH)}</p>
    <p><strong>Current Working Directory:</strong> {os.getcwd()}</p>
    <p><strong>Files in artifacts_yield:</strong> {os.listdir(ART_DIR) if os.path.exists(ART_DIR) else 'Directory not found'}</p>
    </body>
    </html>
    """

# ------------------------------ Results page ------------------------------
@app.post("/predict", response_class=HTMLResponse)
def predict(
    state: str = Form(...),
    district: str = Form(...),
    crop: str = Form(...),
    land_area: float = Form(...),
    end_date: Optional[str] = Form(None)
):
    try:
        out = _predict_core(state, district, crop, land_area, end_date)

        def _f4(x):
            try: return f"{float(x):,.4f}"
            except: return str(x)
        def _f3(x):
            try: return f"{float(x):,.3f}"
            except: return str(x)

        per_area_txt_4 = _f4(out["yield_per_area_pred"])  # tonnes/acre (display)
        total_txt_4  = _f4(out["production_pred"])        # tonnes
        lat_fmt      = _f3(out["lat"]); lon_fmt = _f3(out["lon"])
        acres_disp   = f"{float(out['area_input']):.2f}"

        # Recommendations
        reco = recommend_for_crop(out["crop"])
        irr_title = reco["irrigation"]["title"]
        irr_sub   = reco["irrigation"]["subtitle"]
        fert_npk  = reco["fertilizer_blend"]["npk"]
        fert_note = reco["fertilizer_blend"]["note"]
        pest_active_list = reco.get("pesticides", [])
        pest_grid_items = "".join([f'<div class="pill">{a}</div>' for a in (pest_active_list if pest_active_list else ["No actives"])])

        # Weather rows
        week = out.get("week_features", {}) or {}
        week_rows = "".join(
            f'<div class="kv"><span class="k">{k}</span><span class="v">{_f3(v)}</span></div>'
            for k, v in week.items()
        ) or '<div class="kv"><span class="k">No data</span><span class="v">—</span></div>'

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Analysis Complete</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root{{ --bg:#070b0a; --card: rgba(15, 25, 20, 0.35); --card-border: rgba(255,255,255,0.08);
       --accent:#30d158; --text:#e6f5ea; --muted:#a9b8ae; }}
*{{ box-sizing:border-box; }}
html,body{{ height:100%; margin:0; font-family:"Inter",system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
           background: radial-gradient(1800px 1000px at 60% 0%, #0f1a14 0%, #0b130f 40%, var(--bg) 80%) no-repeat, var(--bg); color:var(--text); }}
.container{{ min-height:100%; width:100%; padding:40px 24px 60px; }}
.panel{{ width:100%; max-width:1200px; margin: 0 auto; padding: 0 6px; }}

.header{{ text-align:center; margin-bottom: 26px; position: relative; }}
.top-home{{ position:absolute; top:-8px; right:0; }}
.home-btn-small{{ display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:12px; color:#e6f5ea;
                 text-decoration:none; background: rgba(48, 209, 88, 0.12); border:1px solid rgba(48, 209, 88, 0.22); }}

.h-title{{ font-size: clamp(32px, 4.5vw, 58px); font-weight: 800; letter-spacing:.3px; }}
.h-title .accent{{ color: var(--accent); }}
.h-sub{{ margin-top:8px; color:#bcd4c5; }}

.grid-2{{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
.card{{ background: var(--card); border:1px solid var(--card-border); border-radius:18px; padding:22px;
        box-shadow:0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06); }}
.card h3{{ margin:4px 0 12px; font-size:18px; color:#dff2e6; }}
.kv{{ display:flex; justify-content:space-between; gap:12px; padding:10px 0; border-top:1px solid rgba(255,255,255,0.06); }}
.kv:first-of-type{{ border-top:none; }}
.k{{ color:#d0e5da; }} .v{{ color:#bfe9cc; font-weight:600; }}

.yield-big{{ font-size: clamp(28px, 4vw, 46px); font-weight:800; color:#c4ffd2; }}
.yield-sub{{ margin-top:6px; color:#9cc0b1; font-weight:600; }}

.pest-grid{{ margin-top:8px; display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; }}
.pill{{ text-align:center; padding:10px 12px; border-radius:12px; background:rgba(155,125,200,.12);
       border:1px solid rgba(155,125,200,.3); color:#e7d9ff; font-weight:700; }}

.adv-wrap{{ margin-top: 24px; }}
.adv-btn{{ display:inline-flex; align-items:center; gap:10px; padding:12px 16px; border-radius:12px; color:#052d14; font-weight:700;
          background: linear-gradient(90deg, #28d17a, #30d158 45%, #28d17a); border:none; cursor:pointer;
          box-shadow:0 8px 18px rgba(48,209,88,0.25), inset 0 1px 0 rgba(255,255,255,0.35); }}
.adv-card{{ margin-top:14px; display:none; }}
.adv-grid{{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
@media (max-width: 900px){{ .grid-2{{ grid-template-columns:1fr; }} .adv-grid{{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
  <div class="container">
    <div class="panel">

      <div class="header">
        <a class="home-btn-small top-home" href="/">🏠 Home</a>
        <div class="h-title">AI <span class="accent">Analysis</span> Complete</div>
        <div class="h-sub">Results for {out["crop"]} prediction in {out["state"]}, {out["district"]}</div>
      </div>

      <div class="grid-2">
        <div class="card">
          <h3>Summary</h3>
          <div class="kv"><span class="k">State</span><span class="v">{out["state"]}</span></div>
          <div class="kv"><span class="k">District</span><span class="v">{out["district"]}</span></div>
          <div class="kv"><span class="k">Crop</span><span class="v">{out["crop"]}</span></div>
          <div class="kv"><span class="k">Area (acres)</span><span class="v">{acres_disp}</span></div>
        </div>

        <div class="card">
          <h3>Predicted {out["crop"]} yield</h3>
          <div class="yield-big">{per_area_txt_4} tonnes/acre</div>
          <div class="yield-sub">Estimated total (model): {total_txt_4} tonnes</div>
        </div>
      </div>

      <!-- Recommendation Cards -->
      <div class="grid-2" style="margin-top:18px;">
        <div class="card" style="background:linear-gradient(180deg, rgba(7,20,15,.6), rgba(7,20,15,.45)); border-color:rgba(48,209,88,0.18);">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:44px;height:44px;border-radius:12px;background:rgba(48,209,88,.12);display:flex;align-items:center;justify-content:center;border:1px solid rgba(48,209,88,.25);">💧</div>
            <h3 style="margin:0;">Irrigation Strategy</h3>
          </div>
          <div class="yield-big" style="margin-top:14px;">{irr_title}</div>
          <div class="yield-sub">{irr_sub}</div>
        </div>

        <div class="card" style="background:linear-gradient(180deg, rgba(36,28,5,.62), rgba(30,24,4,.45)); border-color:rgba(255,198,69,0.22);">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:44px;height:44px;border-radius:12px;background:rgba(255,198,69,.12);display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,198,69,.3);">🧪</div>
            <h3 style="margin:0;">Fertilizer Blend</h3>
          </div>
          <div class="yield-big" style="margin-top:14px;">{fert_npk}</div>
          <div class="yield-sub">{fert_note}</div>
        </div>
      </div>

      <!-- Pesticides -->
      <div class="card" style="margin-top:18px;background:linear-gradient(180deg, rgba(18,7,32,.55), rgba(18,7,32,.45)); border-color:rgba(155,125,200,0.18);">
        <div style="display:flex;align-items:center;gap:12px;">
          <div style="width:44px;height:44px;border-radius:12px;background:rgba(155,125,200,.12);display:flex;align-items:center;justify-content:center;border:1px solid rgba(155,125,200,.3);">🛡️</div>
          <h3 style="margin:0;">Recommended Pesticides</h3>
        </div>
        <div class="pest-grid">{pest_grid_items}</div>
      </div>

      <!-- Advanced insights -->
      <div class="adv-wrap">
        <button class="adv-btn" id="advToggle" aria-expanded="false">Advanced insights</button>
        <div class="card adv-card" id="advCard" aria-hidden="true" style="display:none;">
          <div class="adv-grid">
            <div class="card">
              <h3>Model Context</h3>
              <div class="kv"><span class="k">Latitude</span><span class="v">{lat_fmt}</span></div>
              <div class="kv"><span class="k">Longitude</span><span class="v">{lon_fmt}</span></div>
              <div class="kv"><span class="k">End Date</span><span class="v">{out.get("end_date_used") or "-"}</span></div>
            </div>
            <div class="card">
              <h3>Previous Weeks' Weather Data</h3>
              {week_rows}
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
<script>
  document.addEventListener('DOMContentLoaded', function () {{
    // Bias select to open downward by ensuring viewport room
    const cropSel = document.getElementById('crop');
    if (cropSel) {{
      cropSel.addEventListener('focus', function() {{
        const rect = cropSel.getBoundingClientRect();
        const spaceBelow = window.innerHeight - rect.bottom;
        if (spaceBelow < 220) {{
          window.scrollBy({{ top: 220 - spaceBelow + 20, behavior: 'smooth' }});
        }}
      }});
    }}

    // Advanced insights toggle (robust to repeated clicks and initial state)
    var advBtn = document.getElementById('advToggle');
    var advCard = document.getElementById('advCard');
    if (!advBtn || !advCard) return;
    if (!advCard.style.display) advCard.style.display = 'none';
    advBtn.addEventListener('click', function () {{
      var show = advCard.style.display !== 'block';
      advCard.style.display = show ? 'block' : 'none';
      advBtn.textContent = show ? 'Hide advanced insights' : 'Advanced insights';
      advCard.setAttribute('aria-hidden', show ? 'false' : 'true');
      advBtn.setAttribute('aria-expanded', show ? 'true' : 'false');
    }});
  }});
</script>


</body>
</html>
"""
        return HTMLResponse(html)
    except Exception as e:
    # Use HTMLResponse for browser form submission errors
      return HTMLResponse(
        f"""
        <html>
        <head><title>Error</title></head>
        <body style='font-family:Arial;background:#fff;color:#333;padding:32px'>
        <h2 style='color:#900;'>An error occurred:</h2>
        <pre>{str(e)}</pre>
        <a href='/'>Back to form</a>
        </body>
        </html>
        """,
        status_code=400
    )


@app.post("/predict/", response_class=HTMLResponse)
def predict_trailing(
    state: str = Form(...),
    district: str = Form(...),
    crop: str = Form(...),
    land_area: float = Form(...),
    end_date: Optional[str] = Form(None)
):
    return predict(state, district, crop, land_area, end_date)
