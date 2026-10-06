FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HOME=/home/user

# Hugging Face Spaces runs containers as uid 1000; use the same user everywhere.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# requirements.txt points pip at the CPU-only torch wheels (no CUDA libraries).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the app runs with no internet.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
# var/ holds the SQLite files and must be writable by the runtime user.
RUN mkdir -p /app/var && chown -R user:user /app/var /opt/hf-cache
USER user

EXPOSE 8000 8501
CMD ["python", "run.py"]
