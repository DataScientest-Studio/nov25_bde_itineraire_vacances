#!/bin/bash

set -e

echo "=== Compilation OSRM France (4 régions) ==="

# Créer le dossier data
mkdir -p data
cd data

# Vérifier si déjà compilé
if [ -f "france-3regions-car.osrm" ] && \
   [ -f "france-3regions-bike.osrm" ] && \
   [ -f "france-3regions-foot.osm" ]; then
    echo "✓ Données OSRM déjà compilées, exit..."
    exit 0
fi

# Téléchargement
echo ""
echo "Étape 1: Téléchargement des régions..."
[ ! -f "ile-de-france-latest.osm.pbf" ] && wget --quiet https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf && echo "  ✓ Île-de-France"
[ ! -f "bretagne-latest.osm.pbf" ] && wget --quiet https://download.geofabrik.de/europe/france/bretagne-latest.osm.pbf && echo "  ✓ Bretagne"
[ ! -f "auvergne-latest.osm.pbf" ] && wget --quiet https://download.geofabrik.de/europe/france/auvergne-latest.osm.pbf && echo "  ✓ Auvergne"
[ ! -f "rhone-alpes-latest.osm.pbf" ] && wget --quiet https://download.geofabrik.de/europe/france/rhone-alpes-latest.osm.pbf && echo "  ✓ Rhône-Alpes"

# Fusion
echo ""
echo "Étape 2: Fusion des régions..."
if [ ! -f "france-3regions.osm.pbf" ]; then
    osmium merge ile-de-france-latest.osm.pbf bretagne-latest.osm.pbf auvergne-latest.osm.pbf rhone-alpes-latest.osm.pbf -o france-3regions.osm.pbf
    echo "  ✓ Fusion terminée"
fi

# Copies
echo "Étape 3: Création des copies..."
cp france-3regions.osm.pbf france-3regions-car.osm.pbf
cp france-3regions.osm.pbf france-3regions-bike.osm.pbf
cp france-3regions.osm.pbf france-3regions-foot.osm.pbf


echo ""
echo "=== ✓ Préparation terminée! Les fichiers .pbf sont prêts pour compilation ==="