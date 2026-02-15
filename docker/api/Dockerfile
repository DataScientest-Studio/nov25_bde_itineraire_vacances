FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Gérard Pham"
LABEL description="API FastAPI pour le calcul d'itinéraires de vacances"
LABEL version="1.0"

RUN mkdir -p /data

WORKDIR /app

# Installation de curl pour le healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY app/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copie du code source de l'API
COPY app/ .

# Configuration des variables d'environnement
ENV PYTHONPATH=/app
ENV OSRM_URL_CAR=${OSRM_URL_CAR:-http://osrm-car:5000}
ENV OSRM_URL_BIKE=${OSRM_URL_BIKE:-http://osrm-bike:5000}
ENV OSRM_URL_FOOT=${OSRM_URL_FOOT:-http://osrm-foot:5000}
ENV POSTGRES_HOST=${POSTGRES_HOST:-postgres-vacances}
ENV POSTGRES_PORT=${POSTGRES_PORT:-5432}
ENV POSTGRES_DB=${POSTGRES_DB:-vacances}
ENV POSTGRES_USER=${POSTGRES_USER:-vacances_user}
ENV POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-vacances_password}

# Exposition du port FastAPI
EXPOSE 8000

# Vérification du fonctionnement du service api
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=10s \
    CMD curl --fail http://localhost:8000/health || exit 1

# Commande de démarrage avec rechargement pour le développement
CMD ["uvicorn", "app.main_minimal:app", "--host", "0.0.0.0", "--port", "8000"]