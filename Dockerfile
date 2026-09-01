# EdgeFund -- single container running the scheduler and the dashboard.
#
# Node is present only so the Claude Code CLI can run for the three daily brain
# calls. The trading engine itself is pure Python and keeps working if that
# binary is missing or unauthenticated, so the image stays useful either way.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=America/New_York

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates gnupg tzdata \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @anthropic-ai/claude-code \
 && apt-get purge -y gnupg \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edgefund/ ./edgefund/
COPY scripts/ ./scripts/

# SQLite lives on a mounted volume so history survives redeploys.
RUN mkdir -p /app/data_store
ENV DB_PATH=data_store/edgefund.db
VOLUME ["/app/data_store"]

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=5).status==200 else 1)"

CMD ["uvicorn", "edgefund.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
