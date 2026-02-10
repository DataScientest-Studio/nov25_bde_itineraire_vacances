#!/bin/bash

# Script d'installation et configuration OSRM avec merge de régions
# Régions: Auvergne-Rhône-Alpes, Bretagne, Île-de-France
# Profils: foot, bike, car

set -e

echo "=== Configuration OSRM Multi-Régions ==="

# Créer le dossier principal d'abord
mkdir -p osrm-data
cd osrm-data

# Créer la structure de sous-dossiers
mkdir -p raw merged profiles

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
    echo "  - Téléchargement de $name..."
    
    OUTPUT_FILE="raw/$name-latest.osm.pbf"
    CHECKSUM_FILE="raw/$name-latest.osm.pbf.md5"
    
    # Télécharger le fichier avec reprise si interrompu
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "    Téléchargement en cours (avec reprise automatique)..."
        wget -c --show-progress --progress=bar:force \
             --timeout=30 --tries=5 --retry-connrefused \
             -O "$OUTPUT_FILE" "$url"
        
        # Vérifier que le fichier n'est pas vide ou corrompu
        if [ ! -s "$OUTPUT_FILE" ]; then
            echo "    ERREUR: Fichier vide, suppression..."
            rm -f "$OUTPUT_FILE"
            exit 1
        fi
        
        # Validation basique du fichier PBF
        echo "    Validation du fichier PBF..."
        if ! osmium fileinfo "$OUTPUT_FILE" > /dev/null 2>&1; then
            echo "    ERREUR: Fichier PBF corrompu, suppression..."
            rm -f "$OUTPUT_FILE"
            echo "    Veuillez relancer le script pour télécharger à nouveau."
            exit 1
        fi
        
        echo "    ✓ Fichier valide!"
    else
        echo "    Vérification du fichier existant..."
        if ! osmium fileinfo "$OUTPUT_FILE" > /dev/null 2>&1; then
            echo "    ERREUR: Fichier existant corrompu, suppression..."
            rm -f "$OUTPUT_FILE"
            echo "    Relancez le script pour télécharger à nouveau."
            exit 1
        fi
        echo "    ✓ Fichier existant valide, skip téléchargement."
    fi
done

echo ""
echo "Étape 2: Installation d'osmium-tool pour le merge..."
# Vérifier si osmium est installé
if ! command -v osmium &> /dev/null; then
    echo "  Installation d'osmium-tool..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y osmium-tool
    else
        echo "  ERREUR: apt-get non trouvé. Installez osmium-tool manuellement."
        exit 1
    fi
else
    echo "  osmium-tool déjà installé."
fi

echo ""
echo "Étape 3: Fusion des 3 régions..."
MERGED_OUTPUT="merged/france-regions-merged.osm.pbf"

if [ ! -f "$MERGED_OUTPUT" ]; then
    echo "  Fusion en cours (cela peut prendre 5-10 minutes)..."
    osmium merge \
        raw/auvergne-latest.osm.pbf \
        raw/rhone-alpes-latest.osm.pbf \
        raw/bretagne-latest.osm.pbf \
        raw/ile-de-france-latest.osm.pbf \
        -o "$MERGED_OUTPUT" \
        --overwrite
    
    # Vérifier le fichier mergé
    echo "  Validation du fichier mergé..."
    if ! osmium fileinfo "$MERGED_OUTPUT" > /dev/null 2>&1; then
        echo "  ERREUR: Le fichier mergé est corrompu!"
        rm -f "$MERGED_OUTPUT"
        exit 1
    fi
    
    echo "  ✓ Merge terminé avec succès: france-regions-merged.osm.pbf"
    osmium fileinfo "$MERGED_OUTPUT" | grep -E "(File size|Number of)"
else
    echo "  Vérification du fichier mergé existant..."
    if ! osmium fileinfo "$MERGED_OUTPUT" > /dev/null 2>&1; then
        echo "  ERREUR: Fichier mergé existant corrompu, suppression..."
        rm -f "$MERGED_OUTPUT"
        echo "  Relancez le script pour merger à nouveau."
        exit 1
    fi
    echo "  ✓ Fichier mergé existant valide, skip."
fi

echo ""
echo "Étape 4: Extraction et préparation pour chaque profil OSRM..."
MERGED_FILE="../merged/france-regions-merged.osm.pbf"


declare -A PROFILE_FILES=(
  ["car"]="car.lua"
  ["foot"]="foot.lua"
  ["bike"]="bicycle.lua"
)

for profile in "${PROFILES[@]}"; do
    echo ""
    echo "  === Traitement du profil: $profile ==="
    
    PROFILE_DIR="profiles/$profile"
    mkdir -p "$PROFILE_DIR"
    cd "$PROFILE_DIR"

    LUA_FILE="${PROFILE_FILES[$profile]}"
    
    # Copier le fichier mergé dans le dossier du profil
    if [ ! -f "france-merged.osm.pbf" ]; then
        echo "    Copie du fichier mergé..."
        cp "$MERGED_FILE" france-merged.osm.pbf
    else
        echo "    Fichier OSM déjà présent."
    fi
    
    echo "    1. Extraction des données (osrm-extract)..."
    if [ ! -f "france-merged.osrm" ]; then
        docker run --rm \
            -v "$(pwd):/data" \
            ghcr.io/project-osrm/osrm-backend \
            osrm-extract -p /opt/${LUA_FILE} /data/france-merged.osm.pbf
    else
        echo "       Extraction déjà effectuée."
    fi
    
    echo "    2. Partitionnement (osrm-partition)..."
    if [ ! -f "france-merged.osrm.cells" ]; then
        docker run --rm \
            -v "$(pwd):/data" \
            ghcr.io/project-osrm/osrm-backend \
            osrm-partition /data/france-merged.osrm
    else
        echo "       Partitionnement déjà effectué."
    fi
    
    echo "    3. Personnalisation (osrm-customize)..."
    if [ ! -f "france-merged.osrm.hsgr" ]; then
        docker run --rm \
            -v "$(pwd):/data" \
            ghcr.io/project-osrm/osrm-backend \
            osrm-customize /data/france-merged.osrm
    else
        echo "       Personnalisation déjà effectuée."
    fi
    
    echo "    ✓ Profil $profile prêt!"
    
    cd ../..
done

echo ""
echo "=== Configuration terminée! ==="
echo ""
echo "Pour démarrer un serveur OSRM pour un profil:"
echo ""
for profile in "${PROFILES[@]}"; do
    port=$((5000 + $(echo "$profile" | wc -c)))
    case $profile in
        foot) port=5001 ;;
        bike) port=5002 ;;
        car)  port=5000 ;;
    esac
    
    echo "# Profil $profile (port $port):"
    echo "docker run -d --name osrm-$profile -p $port:5000 \\"
    echo "  -v \"\$(pwd)/profiles/$profile:/data\" \\"
    echo "  ghcr.io/project-osrm/osrm-backend \\"
    echo "  osrm-routed --algorithm mld /data/france-merged.osrm"
    echo ""
done

echo "Exemple de requête (remplacer le port selon le profil):"
echo "curl 'http://localhost:5000/route/v1/driving/4.845020,45.763723;4.890185,45.769835?overview=false'"
echo ""
echo "Taille des fichiers générés:"
du -sh merged/france-regions-merged.osm.pbf 2>/dev/null || echo "  (en cours de génération)"
for profile in "${PROFILES[@]}"; do
    du -sh profiles/$profile/*.osrm* 2>/dev/null | head -3 || true
done