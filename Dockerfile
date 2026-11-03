FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RANKFORGE_DB=/data/events.db
WORKDIR /app
COPY pyproject.toml README.md ./
COPY rankforge ./rankforge
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir . \
    && useradd --uid 10001 --create-home worker \
    && mkdir /data && chown worker:worker /data
COPY artifacts ./artifacts
USER worker
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["python", "-m", "rankforge.cli", "serve", "--host", "0.0.0.0"]
