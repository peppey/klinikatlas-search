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

# Abhängigkeiten exakt aus uv.lock, ohne CUDA-Pakete; PyTorch stattdessen als CPU-Version
# --no-deps verhindert, dass pip-Abhängigkeiten (z. B. von flair) wieder torch und CUDA nachziehen
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project > requirements-all.txt \
 && TORCH_VERSION=$(grep -m1 -oE '^torch==[0-9.]+' requirements-all.txt | cut -d= -f3) \
 && grep -v -E '^(torch|triton|nvidia-|cuda-)' requirements-all.txt > requirements.txt \
 && uv venv \
 && uv pip install --no-deps -r requirements.txt \
 && uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

COPY src ./src
RUN uv pip install --no-deps -e .

# Modelle beim Bauen laden, damit der Start schnell ist
RUN python -c "from flair.models import SequenceTagger; SequenceTagger.load('flair/ner-german-large')" \
 && python -c "from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification; [c.from_pretrained('permediq/SapBERT-DE') for c in (AutoModel, AutoTokenizer)]; [c.from_pretrained('HUMADEX/german_medical_ner') for c in (AutoModelForTokenClassification, AutoTokenizer)]"

EXPOSE 8080
CMD marimo run --include-code --headless --host 0.0.0.0 --port ${PORT} src/prototype.py
