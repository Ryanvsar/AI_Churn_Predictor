# Optional self-hosted server: same UI, plus live database URLs (PostgreSQL, MySQL, ...) and
# analysis on the server instead of in the browser.
#   docker build -t churn-predictor .
#   docker run -p 8000:8000 churn-predictor     -> http://localhost:8000
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# xgboost-cpu is the same library without the multi-GB GPU dependencies.
RUN grep -vE '^(xgboost|pytest|httpx)' requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt xgboost-cpu "psycopg[binary]" pymysql

COPY backend ./backend
COPY site ./site
COPY mappings ./mappings

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
