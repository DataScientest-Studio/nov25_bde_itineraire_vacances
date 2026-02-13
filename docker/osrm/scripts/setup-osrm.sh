#!/bin/bash

# Script d'installation et configuration OSRM avec merge de régions
# Régions: Auvergne, Rhône-Alpes, Bretagne, Île-de-France
# Profils: foot, bike, car

set -euo pipefail

echo "=== Configuration OSRM Multi-Régions ==="

# Détection du répertoire d'exécution (idempotence)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
OSRM_DIR="$BASE_DIR/osrm-data"

mkdir -p "$OSRM_DIR"/{raw,merged,profiles}
cd "$OSRM_DIR"

# URLs des données OSM (Geofabrik)
REGIONS=(
    "auvergne:https://download.geofabrik.de/europe/france/auvergne-latest.osm.pbf"
    "rhone-alpes:https://download.geofabrik.de/europe/france/rhone-alpes-latest.osm.pbf"
    "bretagne:https://download.geofabrik.de/europe/france/bretagne-latest.osm.pbf"
    "ile-de-france:https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf"
)

PROFILES=("foot" "bike" "car")

echo ""
echo "Étape 1: Téléchargement des régions..."
for region_data in "${REGIONS[@]}"; do
    IFS=':' read -r name url <<< "$region_data"

    echo "  - Région: $name"
    OUTPUT_FILE="raw/$name-latest.osm.pbf"

    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "    Téléchargement..."
        wget -c --show-progress --progress=bar:force \
             --timeout=30 --tries=5 --retry-connrefused \
             -O "$OUTPUT_FILE" "$url"

        if [ ! -s "$OUTPUT_FILE" ]; then
            echo "    ERREUR: Fichier vide, suppression."
            rm -f "$OUTPUT_FILE"
            exit 1
        fi

        echo "    Validation..."
        if ! osmium fileinfo "$OUTPUT_FILE" >/dev/null 2>&1; then
            echo "    ERREUR: Fichier corrompu."
            rm -f "$OUTPUT_FILE"
            exit 1
        fi

        echo "    ✓ OK"
    else
        echo "    Fichier déjà présent, validation..."
        if ! osmium fileinfo "$OUTPUT_FILE" >/dev/null 2>&1; then
            echo "    ERREUR: Fichier existant corrompu."
            rm -f "$OUTPUT_FILE"
            exit 1
        fi
        echo "    ✓ OK"
    fi
done

echo ""
echo "Étape 2: Vérification osmium-tool..."
if ! command -v osmium >/dev/null 2>&1; then
    echo "  Installation d'osmium-tool..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y osmium-tool
    else
        echo "  ERREUR: apt-get non disponible."
        exit 1
    fi
else
    echo "  osmium-tool déjà installé."
fi

echo ""
echo "Étape 3: Fusion des régions..."
MERGED_OUTPUT="merged/france-regions-merged.osm.pbf"

if [ ! -f "$MERGED_OUTPUT" ]; then
    echo "  Fusion..."
    osmium merge \
        raw/auvergne-latest.osm.pbf \
        raw/rhone-alpes-latest.osm.pbf \
        raw/bretagne-latest.osm.pbf \
        raw/ile-de-france-latest.osm.pbf \
        -o "$MERGED_OUTPUT" --overwrite

    echo "  Validation..."
    if ! osmium fileinfo "$MERGED_OUTPUT" >/dev/null 2>&1; then
        echo "  ERREUR: Merge corrompu."
        rm -f "$MERGED_OUTPUT"
        exit 1
    fi

    echo "  ✓ Merge OK"
else
    echo "  Fichier mergé déjà présent, validation..."
    if ! osmium fileinfo "$MERGED_OUTPUT" >/dev/null 2>&1; then
        echo "  ERREUR: Fichier mergé corrompu."
        rm -f "$MERGED_OUTPUT"
        exit 1
    fi
    echo "  ✓ OK"
fi

echo ""
echo "Étape 4: Préparation des profils OSRM..."

declare -A PROFILE_FILES=(
  ["car"]="car.lua"
  ["foot"]="foot.lua"
  ["bike"]="bicycle.lua"
)

for profile in "${PROFILES[@]}"; do
    echo ""
    echo "  === Profil: $profile ==="

    PROFILE_DIR="$OSRM_DIR/profiles/$profile"
    mkdir -p "$PROFILE_DIR"

    cd "$PROFILE_DIR"

    LUA_FILE="${PROFILE_FILES[$profile]}"

    # Correction du chemin vers le fichier mergé
    if [ ! -f "france-merged.osm.pbf" ]; then
        echo "    Copie du fichier mergé..."
        cp "$OSRM_DIR/merged/france-regions-merged.osm.pbf" france-merged.osm.pbf
    fi

    echo "    1. osrm-extract..."
    if [ ! -f "france-merged.osrm" ]; then
        docker run --rm \
            -v "$(pwd):/data" \
            ghcr.io/project-osrm/osrm-backend \
            osrm-extract -p "/opt/$LUA_FILE" /data/france-merged.osm.pbf
    fi

    echo "    2. osrm-partition..."
    if [ ! -f "france-merged.osrm.cells" ]; then
        docker run --rm \
            -v "$(pwd):/data" \
            ghcr.io/project-osrm/osrm-backend \
            osrm-partition /data/france-merged.osrm
    fi

    echo "    3. osrm-customize..."
    if [ ! -f "france-merged.osrm.hsgr" ]; then
        docker run --rm \
            -v "$(pwd):/data" \
            ghcr.io/project-osrm/osrm-backend \
            osrm-customize /data/france-merged.osrm
    fi

    echo "    ✓ Profil $profile prêt."

    cd "$OSRM_DIR"
done

echo ""
echo "=== Configuration terminée ==="
echo ""
echo "Commandes pour lancer les serveurs OSRM :"
echo ""

for profile in "${PROFILES[@]}"; do
    case $profile in
        car)  port=5000 ;;
        foot) port=5001 ;;
        bike) port=5002 ;;
    esac

    echo "# Profil $profile (port $port)"
    echo "docker run -d --name osrm-$profile -p $port:5000 \\"
    echo "  -v \"$OSRM_DIR/profiles/$profile:/data\" \\"
    echo "  ghcr.io/project-osrm/osrm-backend \\"
    echo "  osrm-routed --algorithm mld /data/france-merged.osrm"
    echo ""
done

echo "Exemple de requête:"
echo "curl 'http://localhost:5000/route/v1/driving/4.845020,45.763723;4.890185,45.769835?overview=false'"
echo ""
echo "Taille des fichiers générés:"
du -sh merged/france-regions-merged.osm.pbf 2>/dev/null || true
for profile in "${PROFILES[@]}"; do
    du -sh profiles/$profile/*.osrm* 2>/dev/null | head -3 || true
done