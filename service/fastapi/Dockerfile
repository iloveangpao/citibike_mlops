FROM python:3.11-slim

RUN pip install fastapi uvicorn[standard] boto3 joblib scikit-learn==1.7.1
WORKDIR /app
COPY app.py .

# Create a writable models dir and drop privileges
RUN adduser --disabled-password --gecos '' appuser \
 && mkdir -p /app/models \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","2"]
