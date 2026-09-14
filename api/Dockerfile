# Hugging Face Spaces (Docker SDK, free CPU tier). Build context = a staging
# folder containing api/ and data/grid/ - see api/deploy_hf.py.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY --chown=user api/ api/
COPY --chown=user data/ data/

USER user
ENV AEROGAP_DATA=/app/data/grid
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
