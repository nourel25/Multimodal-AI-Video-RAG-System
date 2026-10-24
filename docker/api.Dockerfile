# API and worker image (same code, different command). Build context: repository root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
COPY src/requirements /app/requirements
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements/base.txt -r requirements/ml.txt

COPY src /app

RUN useradd --create-home app \
 && mkdir -p /app/assets /models \
 && chown -R app:app /app/assets /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
