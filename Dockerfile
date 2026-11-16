# Wetland multi-station observation service.
#
# One image serves the API, the simulator and the verifier; the compose file
# starts PostgreSQL and the API.  Everything is fictional test data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# psycopg[binary] ships wheels, so no build toolchain is needed.
COPY requirements.txt ./
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY app ./app
COPY simulator ./simulator
COPY verifier ./verifier
COPY scripts ./scripts
COPY samples ./samples

RUN useradd --create-home --uid 10001 wetland
USER wetland

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
