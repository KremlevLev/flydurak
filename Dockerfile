FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install .
COPY run.py ./
EXPOSE 8765
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
