import os, json, time, threading, tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import joblib
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ---- configuration from env
BUCKET = os.getenv("MODEL_BUCKET", "citibike-models")
KEY    = os.getenv("MODEL_KEY", "inference/latest.pkl")
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "devtoken")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
MODEL_PATH = os.path.join(MODEL_DIR, "latest.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# path-style is robust for MinIO/custom endpoints
boto_cfg = Config(s3={'addressing_style': 'path'})  # auto/virtual/path supported by botocore
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=boto_cfg,
)

class PredictIn(BaseModel):
    station_id: int
    timestamp: str   # ISO 8601

class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._etag: Optional[str] = None
        self._last_modified: Optional[float] = None
        self._last_loaded_at: Optional[float] = None

    def info(self):
        return {
            "has_model": self._model is not None,
            "etag": self._etag,
            "last_modified": self._last_modified,
            "last_loaded_at": self._last_loaded_at,
            "bucket": BUCKET, "key": KEY
        }

    def _download_to(self, dst_path: str):
        # download to a temp file first, then atomic replace
        fd, tmp_path = tempfile.mkstemp(dir=MODEL_DIR, suffix=".tmp")
        os.close(fd)
        try:
            s3.download_file(BUCKET, KEY, tmp_path)                # managed transfer
            os.replace(tmp_path, dst_path)                         # atomic swap on POSIX/Windows
        except Exception:
            # cleanup and re-raise
            try: os.remove(tmp_path)
            except Exception: pass
            raise

    def _head(self):
        return s3.head_object(Bucket=BUCKET, Key=KEY)              # to compare ETag / LastModified

    def ensure_loaded(self) -> bool:
        """If not loaded, try once to load; return True if present."""
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            try:
                meta = self._head()
                self._download_to(MODEL_PATH)
                self._model = joblib.load(MODEL_PATH)
                self._etag = meta.get("ETag", "").strip('"')
                lm = meta.get("LastModified")
                self._last_modified = lm.timestamp() if lm else None
                self._last_loaded_at = time.time()
                return True
            except ClientError as e:
                # 404 -> model not published yet
                if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                    return False
                raise

    def force_reload(self) -> dict:
        """Always check remote and reload if different ETag (or missing)."""
        with self._lock:
            try:
                meta = self._head()
                new_etag = meta.get("ETag", "").strip('"')
                if self._etag != new_etag or self._model is None:
                    self._download_to(MODEL_PATH)
                    self._model = joblib.load(MODEL_PATH)
                    self._etag = new_etag
                    lm = meta.get("LastModified")
                    self._last_modified = lm.timestamp() if lm else None
                    self._last_loaded_at = time.time()
                    return {"reloaded": True, "etag": self._etag}
                return {"reloaded": False, "etag": self._etag}
            except ClientError as e:
                # propagate meaningful message
                code = e.response.get("Error", {}).get("Code")
                raise HTTPException(status_code=503, detail=f"Model not available: {code}")

    def predict(self, station_id: int, ts_iso: str) -> float:
        if self._model is None and not self.ensure_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded; try again later.")
        # toy features: weekday & hour derived from timestamp string
        from datetime import datetime
        t = datetime.fromisoformat(ts_iso)
        x = [[station_id, t.weekday(), t.hour]]
        return float(self._model.predict(x)[0])

mgr = ModelManager()
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "model": mgr.info()}

@app.post("/admin/reload")
def admin_reload(x_auth_token: str = Header(default="")):
    if x_auth_token != RELOAD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token.")
    return mgr.force_reload()

@app.post("/predict")
def predict(body: PredictIn):
    y = mgr.predict(body.station_id, body.timestamp)
    return {"bikes_available": int(round(y))}

