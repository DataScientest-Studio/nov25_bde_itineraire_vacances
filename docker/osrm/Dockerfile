# =============================================================================
# PROJET ITINERAIRE DE VACANCES - Dockerfile OSRM Multi-Profils
# =============================================================================

FROM osrm/osrm-backend:latest

# Métadonnées
LABEL maintainer="Gérard Pham"
LABEL description="OSRM Backend multi-profils pour les régions Auvergne Rhône-Alpes, Île-de-France, Bretagne"
LABEL version="1.0"

# Installation des utilitaires supplémentaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    osmium-tool \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /data

# Script de téléchargement et préparation des données multi-profils
COPY prepare_osrm_data.sh /prepare_osrm_data.sh
RUN chmod +x /prepare_osrm_data.sh

# Script de démarrage pour gérer plusieurs profils
COPY start_osrm.sh /start_osrm.sh
RUN chmod +x /start_osrm.sh

# Commande par défaut pour démarrer OSRM avec gestion des profils
CMD ["/bin/bash", "/start_osrm.sh"]