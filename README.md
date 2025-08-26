# Citibike MLOps Pipeline

A production-grade MLOps pipeline for predicting Citibike demand using Airflow, DVC, MinIO, PostgreSQL, and FastAPI.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
  - [Environment Configuration](#environment-configuration)
  - [Docker Setup](#docker-setup)
  - [Database Initialization](#database-initialization)
  - [Airflow Configuration](#airflow-configuration)
  - [MinIO Setup](#minio-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker and Docker Compose v2
- Git
- Python 3.12+
- DVC
- MinIO client (mc)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/citibike_mlops.git
cd citibike_mlops
```

2. Create and configure .env file:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Start the services:
```bash
docker compose up -d
```

4. Initialize the database:
```bash
docker exec -it citibike_mlops-postgres-1 psql -U airflow -d airflow -f /db_init/schema.sql
```

5. Access Airflow UI at http://localhost:8080 (default credentials: airflow/airflow)

## Detailed Setup

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Airflow
AIRFLOW_UID=50000
AIRFLOW_GID=50000
AIRFLOW_PROJ_DIR=./

# PostgreSQL
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_ENDPOINT_URL=http://minio:9000

# API
FASTAPI_RELOAD_TOKEN=your_secure_token_here
MODEL_BUCKET=citibike-models
MODEL_KEY=inference/latest.pkl
```

### Docker Setup

1. Build the images:
```bash
docker compose build --no-cache
```

2. Start the services:
```bash
docker compose up -d
```

3. Verify all services are running:
```bash
docker compose ps
```

Expected services:
- airflow-webserver
- airflow-scheduler
- airflow-worker
- postgres
- minio
- fastapi

### Database Initialization

1. Wait for PostgreSQL to be ready:
```bash
docker compose logs -f postgres
```

2. Run the schema initialization:
```bash
docker exec -it citibike_mlops-postgres-1 psql -U airflow -d airflow -f /db_init/schema.sql
```

3. Verify tables are created:
```bash
docker exec -it citibike_mlops-postgres-1 psql -U airflow -d airflow -c "\dt"
```

### Airflow Configuration

1. Access Airflow UI at http://localhost:8080

2. Add required connections:
   - Admin → Connections → Add new record
   
   a. PostgreSQL Connection:
   ```
   Conn Id: POSTGRES_CONN
   Conn Type: Postgres
   Host: postgres
   Schema: airflow
   Login: airflow
   Password: airflow
   Port: 5432
   ```

   b. FastAPI Connection:
   ```
   Conn Id: FASTAPI
   Conn Type: HTTP
   Host: http://fastapi
   Port: 8080
   ```

3. Add Variables:
   - Admin → Variables → Add new record
   ```
   Key: baseline_rmse
   Value: 9999
   ```
   ```
   Key: INFERENCE_RELOAD_TOKEN
   Value: your_secure_token_here  # Same as FASTAPI_RELOAD_TOKEN in .env
   ```

### MinIO Setup

1. Access MinIO Console at http://localhost:9001
   - Username: minioadmin
   - Password: minioadmin

2. Create required buckets:
   ```bash
   # Install MinIO client (mc)
   mc alias set myminio http://localhost:9000 minioadmin minioadmin
   mc mb myminio/citibike-models
   ```

3. Configure bucket access:
   - Enable read/write access for Airflow
   - Set up versioning for model artifacts

## Running the Pipeline

1. Enable the DAG in Airflow UI:
   - Browse to DAGs
   - Find 'citibike_pipeline'
   - Toggle switch to enable

2. Trigger a manual run:
   - Click the Play button → Trigger DAG

3. Monitor execution:
   - Graph View shows task status
   - Logs available for each task

## Monitoring and Maintenance

### Log Monitoring
```bash
# View service logs
docker compose logs -f airflow-worker

# View specific task logs
docker compose logs -f airflow-worker | grep "train_job"
```

### Database Maintenance
```bash
# Connect to PostgreSQL
docker exec -it citibike_mlops-postgres-1 psql -U airflow -d airflow

# Monitor table sizes
SELECT pg_size_pretty(pg_total_relation_size('trips_raw'));
```

### MinIO Maintenance
```bash
# List models
mc ls myminio/citibike-models/inference/

# Clean old versions
mc rm --recursive --force myminio/citibike-models/inference/archive/
```

## Troubleshooting

### Common Issues

1. DVC not found:
```bash
# Check DVC installation in worker
docker exec -it citibike_mlops-airflow-worker-1 which dvc
```

2. Database connection issues:
```bash
# Verify PostgreSQL is running
docker compose ps postgres
```

3. MinIO access problems:
```bash
# Test MinIO connection
mc ls myminio/citibike-models
```

### Debugging Tips

1. Check container logs:
```bash
docker compose logs -f
```

2. Verify environment variables:
```bash
docker exec -it citibike_mlops-airflow-worker-1 env
```

3. Test component connectivity:
```bash
# Test PostgreSQL
docker exec -it citibike_mlops-airflow-worker-1 nc -zv postgres 5432

# Test MinIO
docker exec -it citibike_mlops-airflow-worker-1 nc -zv minio 9000
```

## Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [MinIO Documentation](https://min.io/docs/minio/container/index.html)