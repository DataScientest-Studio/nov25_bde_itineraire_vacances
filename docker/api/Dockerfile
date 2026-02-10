FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Gérard Pham"
LABEL description="API FastAPI pour le calcul d'itinéraires de vacances"
LABEL version="1.0"

RUN mkdir -p /data

WORKDIR /app

# Copie et installation des dépendances Python
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copie du code source de l'API
COPY src/app/ ./app/

# Configuration des variables d'environnement
ENV PYTHONPATH=/app
ENV OSRM_URL=${OSRM_URL:-http://osrm-backend:5000}
ENV POSTGRES_HOST=${POSTGRES_HOST:-postgres-vacances}
ENV POSTGRES_PORT=${POSTGRES_PORT:-5432}
ENV POSTGRES_DB=${POSTGRES_DB:-vacances}
ENV POSTGRES_USER=${POSTGRES_USER:-vacances_user}
ENV POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-vacances_password}

# Exposition du port FastAPI
EXPOSE 8000

# Commande de démarrage avec rechargement pour le développement
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]