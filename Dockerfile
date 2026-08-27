# di-app -- the DocumentIntelligence application image.
#
# One image, several roles: today it runs the evaluation CLI, and the web server and
# worker become additional compose services off this same build. They share a codebase
# and a dependency set, so they must share an image.
#
# The document generator is deliberately NOT here: it needs Chromium and the augraphy
# stack, ~2.4 GB the application has no use for.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DI_DATASET_ROOT=/data \
    DI_REPORTS_DIR=/reports

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY di.toml ./
COPY core/ ./core/
COPY normalize/ ./normalize/
COPY eval/ ./eval/
COPY extract/ ./extract/
# The scoring tests fixture off six committed sample documents. Without them 14
# tests skip-by-crashing rather than failing loudly, which is worse than either.
COPY tools/document-generator/samples/ ./tools/document-generator/samples/
COPY tests/ ./tests/

ENTRYPOINT ["python", "-m", "eval.cli"]
CMD ["--help"]
