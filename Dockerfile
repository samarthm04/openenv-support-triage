FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --system || true

COPY . .

# Ensure dependencies are installed just in case uv sync failed or to install current project
RUN pip install .

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

CMD ["python", "-m", "server.app"]
