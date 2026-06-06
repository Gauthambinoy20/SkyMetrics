# Slim image serving the SkyMetrics API and dashboard.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime deps first for better layer caching.
COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

# Install the package.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Non-root user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501

# Default to the API; compose overrides the command for the dashboard service.
CMD ["uvicorn", "skymetrics.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
