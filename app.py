import os
import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = os.environ.get("MODEL_FILE", "final_pipeline.joblib")
predictor = None

def try_load(path):
    try:
        obj = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        return obj
    except Exception as e:
        logger.exception(f"Failed loading {path}: {e}")
        return None

predictor = try_load(MODEL_PATH)

# fallback: if wrapper not present try to assemble from encoder + raw model
if predictor is None:
    try:
        encoder = joblib.load("encoder.pkl")
        raw_model = joblib.load("xgb_final_model.joblib")
        logger.info("Loaded encoder and raw model; building runtime wrapper.")

        class PredictorWrapper:
            def __init__(self, encoder, model):
                self.encoder = encoder
                self.model = model
                self.cat_cols = list(encoder.feature_names_in_)
                self.expected = list(model.feature_names_in_)

            def prepare(self, raw_df, fallback=True):
                raw_df = raw_df.copy()
                cat_df = raw_df[self.cat_cols].astype(str).copy()
                if fallback:
                    for i, col in enumerate(self.cat_cols):
                        allowed = set(self.encoder.categories_[i])
                        cat_df[col] = cat_df[col].where(cat_df[col].isin(allowed),
                                                        "Unknown" if "Unknown" in allowed else list(self.encoder.categories_[i])[0])
                enc_arr = self.encoder.transform(cat_df)
                enc_df = pd.DataFrame(enc_arr, columns=self.cat_cols, index=raw_df.index)
                rows = []
                for idx, _ in raw_df.iterrows():
                    row = {}
                    for c in self.cat_cols:
                        row[c] = float(enc_df.at[idx, c])
                    for nf in [f for f in self.expected if f not in self.cat_cols]:
                        row[nf] = float(raw_df.at[idx, nf]) if nf in raw_df.columns else 0.0
                    rows.append(row)
                return pd.DataFrame(rows, columns=self.expected)

            def predict(self, raw_df):
                X = self.prepare(raw_df)
                return self.model.predict(X)

        predictor = PredictorWrapper(encoder, raw_model)
    except Exception:
        logger.exception("Runtime wrapper build failed; predictor remains None.")
        predictor = None

@app.get("/")
def health():
    return jsonify({"status": "ok", "model_loaded": predictor is not None})

@app.post("/predict")
def predict_route():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503
    try:
        req = request.get_json(force=True)
        data = req.get("data") if isinstance(req, dict) and "data" in req else req
        if data is None:
            return jsonify({"error": "No JSON body found"}), 400
        df = pd.DataFrame([data])
        preds = predictor.predict(df)
        return jsonify({"prediction": list(map(float, preds))})
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
