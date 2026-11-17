FROM python:3.12-slim

# git wird für die Abhängigkeit de-klinikatlas (GitHub) gebraucht
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app
ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Abhängigkeiten aus uv.lock ohne CUDA-Pakete, dazu PyTorch für CPU
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project \
      | grep -v -E '^(torch|triton|nvidia-|cuda-)' > requirements.txt \
 && uv venv \
 && uv pip install -r requirements.txt \
 && uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.11.0

COPY src ./src
RUN uv pip install --no-deps -e .

# Modelle beim Bauen laden, damit der Start schnell ist
RUN python -c "from flair.models import SequenceTagger; SequenceTagger.load('flair/ner-german-large')" \
 && python -c "from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification; [c.from_pretrained('permediq/SapBERT-DE') for c in (AutoModel, AutoTokenizer)]; [c.from_pretrained('HUMADEX/german_medical_ner') for c in (AutoModelForTokenClassification, AutoTokenizer)]"

COPY notebooks/prototype.py ./notebooks/prototype.py

EXPOSE 8080
CMD marimo run --include-code --headless --host 0.0.0.0 --port ${PORT} notebooks/prototype.py
