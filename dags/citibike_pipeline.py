# from airflow.decorators import dag, task
# from airflow.operators.python import BranchPythonOperator
# from airflow.operators.bash import BashOperator
# from airflow.providers.postgres.hooks.postgres import PostgresHook
# from airflow.models import Variable
# from airflow.providers.http.operators.http import HttpOperator
# import boto3
# import pendulum, requests, json, os, subprocess, pandas as pd, joblib

# DATA_DIR = "/opt/airflow/data"
# S3_BUCKET="citibike-models"

# from airflow.sdk import dag, task, get_current_context  # Airflow 3 imports
# from airflow.models import Variable



# @dag(schedule="0 2 * * *",
#      start_date=pendulum.datetime(2025,1,1,tz="UTC"),
#      catchup=False,
#      tags=["citibike"])
# def citibike_pipeline():

#     @task()
#     def fetch_trip_file():
#         url = f"https://s3.amazonaws.com/tripdata/{pendulum.yesterday().format('YYYYMM')}-citibike-tripdata.csv.zip"
#         local = f"{DATA_DIR}/trips-{pendulum.yesterday().format('YYYYMM')}.zip"
#         with open(local,"wb") as f:
#             f.write(requests.get(url, timeout=120).content)           # public S3 listing :contentReference[oaicite:2]{index=2}
#         return local

#     @task()
#     def load_trips(path:str):
#         df = pd.read_csv(path, compression="zip", low_memory=False)
#         pg = PostgresHook("POSTGRES_CONN").get_conn()
#         df.to_sql("trips_raw", pg, if_exists="append", index=False)

#     @task()
#     def fetch_station_status():
#         feed = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"  # official GBFS feed :contentReference[oaicite:3]{index=3}
#         data = requests.get(feed, timeout=20).json()
#         pg = PostgresHook("POSTGRES_CONN").get_conn()
#         cur = pg.cursor()
#         cur.execute("INSERT INTO station_status_raw VALUES (NOW(), %s)", (json.dumps(data),))

#     @task()
#     def feature_engineering():
#         # simplified for brevity; full logic in ml/features.py
#         return subprocess.run(["python","-m","ml.features","--out",f"{DATA_DIR}/features.parquet"], check=True).returncode

#     @task()
#     def train():
#         return subprocess.check_output(["python","-m","ml.train","--date","{{ ds }}"]).decode()

#     @task.branch(task_id="decide")
#     def decide():
#         ctx = get_current_context()                           # no provide_context needed
#         rmse = ctx["ti"].xcom_pull(task_ids="train_job")
#         baseline = float(Variable.get("baseline_rmse", 9999))
#         return "register" if rmse > baseline * 1.10 else "notify"

#     branch = BranchPythonOperator(task_id="decide", python_callable=decide, provide_context=True)

#     register = BashOperator(
#         task_id="register_model",
#         bash_command="dvc add models/latest.pkl && dvc push",          # DVC→MinIO example :contentReference[oaicite:4]{index=4}
#         env={"AWS_ENDPOINT_URL":os.getenv("AWS_ENDPOINT_URL")}
#     )

#     @task()
#     def publish_model():
#         s3 = boto3.client("s3",
#             endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
#             aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
#         bucket = os.getenv("MODEL_BUCKET", "citibike-models")
#         key    = os.getenv("MODEL_KEY", "inference/latest.pkl")
#         s3.upload_file("models/latest.pkl", bucket, key)

#     publish = publish_model()

#     reload_api = HttpOperator(
#         task_id="reload_inference",
#         http_conn_id="FASTAPI",                  # create this Connection in UI
#         endpoint="/admin/reload",
#         method="POST",
#         headers={"X-Auth-Token": Variable.get("INFERENCE_RELOAD_TOKEN")},
#         log_response=True
#     )

#     build = BashOperator(
#         task_id="build_image",
#         bash_command="docker build -t ghcr.io/$GITHUB_REPOSITORY/infer:{{ ds_nodash }} ./services/fastapi"
#     )

#     notify = BashOperator(
#         task_id="notify_slack",
#         bash_command='curl -X POST -H "Content-type: application/json" --data "{\"text\":\"Citibike DAG finished\"}" $SLACK_WEBHOOK_URL'
#     )

#     fetch_trip_file() >> load_trips() >> fetch_station_status() >> feature_engineering() >> train() >> branch
#     branch >> publish >> reload_api
#     # branch >> notify

# citibike_dag = citibike_pipeline()




# dags/citibike_pipeline.py
from __future__ import annotations

import os
import json
import pendulum
from airflow.sdk import dag, task, get_current_context  # Airflow 3 Task SDK
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.http.operators.http import HttpOperator
from airflow.exceptions import AirflowFailException

# --------- Config (env-driven to match your Compose)
DATA_DIR = "/opt/airflow/data"
TRIP_BASE = "https://s3.amazonaws.com/tripdata"
GBFS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "citibike-models")
MODEL_KEY = os.getenv("MODEL_KEY", "inference/latest.pkl")

# --------- DAG
@dag(
    dag_id="citibike_pipeline",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["citibike", "ml", "dvc", "minio"],
    default_args={"owner": "airflow"},
)
def citibike_pipeline():
    """
    Ingest -> Feature -> Train -> (branch) -> [register to DVC] -> Publish to MinIO -> Tell FastAPI to reload.

    Connections needed:
      - POSTGRES_CONN  (Postgres to optional raw table for station status)
      - FASTAPI        (HTTP base URL, e.g. http://fastapi:8080)

    Variables:
      - INFERENCE_RELOAD_TOKEN  (shared secret for /admin/reload)
    """

    

    @task()
    def ingest_trips(ds: str) -> str:
        """
        Download prior full month's trip archive to /opt/airflow/data and return the local path.
        We try both known filename patterns and fall back one extra month if needed.
        """
        import requests, pendulum, os

        def prev_full_month(datestr: str) -> pendulum.DateTime:
            # first day of ds's month, then step back 1 month -> full previous month
            return pendulum.parse(datestr).start_of("month").subtract(months=1)

        def candidate_urls(yyyymm: str):
            base = "https://s3.amazonaws.com/tripdata"
            return [
                f"{base}/{yyyymm}-citibike-tripdata.csv.zip",
                f"{base}/{yyyymm}-citibike-tripdata.zip",
            ]

        def first_existing_url(yyyymm: str):
            for url in candidate_urls(yyyymm):
                try:
                    h = requests.head(url, allow_redirects=True, timeout=15)
                    if h.status_code == 200:
                        return url
                except requests.RequestException:
                    pass
            return None

        m0 = prev_full_month(ds).format("YYYYMM")
        url = first_existing_url(m0)
        if not url:
            m1 = prev_full_month(ds).subtract(months=1).format("YYYYMM")
            url = first_existing_url(m1)
        if not url:
            raise AirflowFailException(
                f"No tripdata found for {m0} or the prior month in Citi Bike S3 archive."
            )

        os.makedirs(DATA_DIR, exist_ok=True)
        out = f"{DATA_DIR}/trips-{url.split('/')[-1]}"
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        return out

    @task()
    def ingest_station_status() -> None:
        """Append one snapshot of GBFS station status into Postgres (optional raw table)."""
        import requests
        js = requests.get(GBFS_URL, timeout=20).json()
        hook = PostgresHook(postgres_conn_id="POSTGRES_CONN")
        with hook.get_conn() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO station_status_raw VALUES (NOW(), %s)", (json.dumps(js),))
            conn.commit()

    @task()
    def feature_job(trips_zip_path: str) -> str:
        """Generate features from trip data and optionally add weather data."""
        import subprocess, os, sys
        
        # Ensure output directory exists
        out_dir = "/opt/airflow/data"
        os.makedirs(out_dir, exist_ok=True)
        out = f"{out_dir}/features.parquet"
        
        # Debug guard (safe to keep or remove later)
        assert isinstance(trips_zip_path, str), f"not a str: {type(trips_zip_path)}"
        
        # Use system Python since packages are installed at container level
        env = os.environ.copy()
        env["PYTHONPATH"] = "/opt/airflow/src"
        
        _log = print  # Simple logging
        _log(f"Using Python: {sys.executable}")
        _log(f"PYTHONPATH: {env['PYTHONPATH']}")
        
        # Run the feature generation
        cmd = [
            sys.executable,
            "-m", "ml.features",
            "--trips_zip", trips_zip_path,
            "--out", out
        ]
        _log(f"Running command: {' '.join(cmd)}")
        
        subprocess.run(
            cmd,
            check=True,
            env=env,
        )
        return out

    @task()
    def train_job(features_path: str) -> float:
        """
        Train the model; expected to write models/latest.pkl and metrics.json with {"rmse": ...}.
        Returns the RMSE as float to XCom.
        """
        import subprocess, json, os, sys
        
        # Create model directory
        model_dir = "/opt/airflow/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/latest.pkl"
        
        # Use system Python since packages are installed at container level
        env = os.environ.copy()
        env["PYTHONPATH"] = "/opt/airflow/src"
        
        _log = print  # Simple logging
        _log(f"Using Python: {sys.executable}")
        _log(f"PYTHONPATH: {env['PYTHONPATH']}")
        _log(f"Model directory: {model_dir}")
        _log(f"Features path: {features_path}")
        
        # Run the training
        cmd = [
            sys.executable,  # Use system Python
            "-m", "ml.train",
            "--features", str(features_path),
            "--out", str(model_path)
        ]
        _log(f"Running command: {' '.join(cmd)}")
        
        subprocess.run(cmd, check=True, env=env)
        
        # Read metrics from the model directory
        metrics_path = os.path.join(os.path.dirname(model_path), "metrics.json")
        _log(f"Reading metrics from: {metrics_path}")
        with open(metrics_path) as f:
            metrics = json.load(f)
            rmse = float(metrics["rmse"])
            _log(f"Training RMSE: {rmse}")
        return rmse

    @task.branch(task_id="decide")
    def decide(rmse: float) -> str:
        """
        Branch: if RMSE worse than baseline*1.10, register/publish/reload;
        else skip registering (but pipeline still ends cleanly).
        """
        baseline = float(Variable.get("baseline_rmse"))
        return "register" if rmse <= baseline * 0.95 else "skip_register"

    # --- Branch targets
    register = BashOperator(
        task_id="register",
        bash_command="/home/airflow/.local/bin/dvc add models/latest.pkl && /home/airflow/.local/bin/dvc push",
        env={  # ensure DVC sees creds; Compose already provides these for you
            "AWS_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL", ""),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "PATH": "/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin",  # Ensure PATH is set
        },
    )

    skip_register = EmptyOperator(task_id="skip_register")

    @task()
    def publish_model() -> str:
        """Copy the trained artifact to a stable serving key (e.g., inference/latest.pkl)."""
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        s3.upload_file("models/latest.pkl", MODEL_BUCKET, MODEL_KEY)
        return f"s3://{MODEL_BUCKET}/{MODEL_KEY}"

    # Tell FastAPI to hot-reload the model (no rebuild)
    reload_api = HttpOperator(
        task_id="reload_inference",
        http_conn_id="FASTAPI",                # Configure in Airflow UI (base_url like http://fastapi:8080)
        endpoint="/admin/reload",
        method="POST",
        headers={"X-Auth-Token": Variable.get("INFERENCE_RELOAD_TOKEN")},
        log_response=True,
        response_check=lambda r: r.status_code == 200,
    )

    # ---------- Wiring
    trips_zip = ingest_trips()
    _ = ingest_station_status()
    feats = feature_job(trips_zip)
    rmse = train_job(feats)
    branch = decide(rmse)

    # If we "register", also publish to MinIO and ping FastAPI to reload.
    branch >> register
    register >> publish_model() >> reload_api

    # If we do not register, just end.
    branch >> skip_register


dag = citibike_pipeline()

