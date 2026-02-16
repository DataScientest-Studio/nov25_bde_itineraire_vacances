FROM python:3.11-slim

WORKDIR /app

# Dépendances UI
COPY docker/ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copie tout le dossier UI (app + modules + assets)
COPY docker/ui/ /app/

# Pour que les imports (infra/core/services/config/data/...) marchent
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
