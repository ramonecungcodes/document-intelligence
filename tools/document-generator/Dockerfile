# di-document-generator
#
# Builds the synthetic document corpus with no local Python or Chrome needed.
# On-demand only: nothing long-running lives in here.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DI_DATASET_ROOT=/data \
    CHROME_BIN=/usr/bin/chromium \
    CHROME_NO_SANDBOX=1 \
    NUMBA_DISABLE_JIT=1

# chromium            renders the HTML to PDF
# libgl1/libglib2.0-0 augraphy pulls opencv-python, which links against them
# fonts-*             the documents ask for Georgia/Calibri/Cambria/Arial/Courier.
#                     None of those ship on Linux, so install the metric-compatible
#                     substitutes: without them Chromium silently falls back to one
#                     face and the layout variety the corpus depends on collapses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        libgl1 libglib2.0-0 \
        fonts-liberation fonts-dejavu-core \
        fonts-crosextra-carlito fonts-crosextra-caladea \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate.py templates.py bundle.py render_pdfs.py degrade.py cli.py ./

# augraphy writes a texture cache next to the working directory; keep it off the
# mounted volume so it never lands in the dataset.
ENV AUGRAPHY_CACHE=/tmp/augraphy_cache
WORKDIR /tmp

VOLUME ["/data"]
ENTRYPOINT ["python", "/app/cli.py"]
CMD ["help"]
