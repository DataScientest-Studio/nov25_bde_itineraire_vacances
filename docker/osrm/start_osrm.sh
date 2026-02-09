#!/bin/bash

# Script de démarrage OSRM avec support multi-profils

set -e

echo "  Démarrage du service OSRM multi-profils..."

# Préparation des données si nécessaire
/prepare_osrm_data.sh

# Profils disponibles
PROFILES=("car" "bike" "walk")

# Vérifier quel profil utiliser (variable d'environnement ou défaut: car)
PROFILE=${OSRM_PROFILE:-"car"}

if [[ ! " ${PROFILES[@]} " =~ " ${PROFILE} " ]]; then
    echo "  Erreur: Profil '$PROFILE' non valide"
    echo "  Profils disponibles: ${PROFILES[*]}"
    exit 1
fi

echo "  Utilisation du profil: $PROFILE"

# Vérifier si les données du profil existent
if [ ! -f "/data/combined-$PROFILE.osrm" ]; then
    echo "  Erreur: Données du profil '$PROFILE' non trouvées"
    echo "  Fichiers disponibles:"
    ls -la /data/combined-*.osrm 2>/dev/null || echo "    Aucune donnée OSRM trouvée"
    exit 1
fi

echo "  Démarrage d'OSRM avec le profil $PROFILE..."
echo "  Accès au service: http://localhost:5000"
echo "  Profils disponibles: ${PROFILES[*]}"
echo ""
echo "  Pour changer de profil, utilisez la variable d'environnement OSRM_PROFILE"
echo "  Exemple: docker-compose run -e OSRM_PROFILE=bike osrm-multi-regions"

# Démarrage d'OSRM avec le profil sélectionné
exec osrm-routed --algorithm mld /data/combined-$PROFILE.osrm
