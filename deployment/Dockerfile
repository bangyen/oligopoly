FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# No system packages needed: every dependency ships prebuilt wheels

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# Install Python dependencies
RUN pip install --no-cache-dir ".[api]"

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Default: the REST API. For the Scenario Lab dashboard instead:
#   docker run -p 5050:5050 oligopoly python dashboard/main.py
EXPOSE 5050
CMD ["uvicorn", "sim.api:app", "--host", "0.0.0.0", "--port", "8000"]
