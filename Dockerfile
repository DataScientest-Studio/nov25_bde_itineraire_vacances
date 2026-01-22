# =============================================================================
# PROJET ITINÉRAIRE DE VACANCES - Dockerfile Airflow
# =============================================================================

# Image de base officielle Airflow
FROM apache/airflow:2.8.1-python3.11

# Métadonnées
LABEL maintainer="Amadou Adjanouhoun"
LABEL description="Image Airflow Custom avec H3 et PostGIS support"
LABEL version="1.0"

# -----------------------------------------------------------------------------
# 1. Installation des dépendances système (Nécessaire pour GeoSpatial)
# -----------------------------------------------------------------------------
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. Installation des librairies Python via requirements.txt
# -----------------------------------------------------------------------------
USER airflow

# On copie le fichier requirements.txt dans l'image
COPY requirements.txt /requirements.txt

# On installe les librairies
RUN pip install --no-cache-dir -r /requirements.txt

# -----------------------------------------------------------------------------
# 3. Configuration finale
# -----------------------------------------------------------------------------
# Ajout du dossier src au PYTHONPATH pour faciliter les imports
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/src"