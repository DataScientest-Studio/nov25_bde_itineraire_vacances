#!/bin/bash

# Script de préparation des données OSRM pour les régions françaises
# Auvergne Rhône-Alpes, Île-de-France, Bretagne
# Support des profils : car, bike, walk

set -e

echo "  Préparation des données OSRM pour les régions françaises..."

# Liste des régions avec leurs URLs Geofabrik
declare -A REGIONS=(
    ["auvergne-rhone-alpes"]="https://download.geofabrik.de/europe/france/auvergne-rhone-alpes-latest.osm.pbf"
    ["ile-de-france"]="https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf"
    ["bretagne"]="https://download.geofabrik.de/europe/france/bretagne-latest.osm.pbf"
)

# Profils OSRM à générer
PROFILES=("car" "bicyle" "foot")

# Vérifier si les données combinées existent déjà
if [ -f "/data/combined.osrm" ]; then
    echo "  Données OSRM combinées déjà présentes"
    echo "  Démarrage du service OSRM..."
    exit 0
fi

# Téléchargement des données
echo " Téléchargement des données régionales..."
for region in "${!REGIONS[@]}"; do
    pbf_file="${region}-latest.osm.pbf"
    
    if [ ! -f "/data/$pbf_file" ]; then
        echo "   Téléchargement de $region..."
        wget -O "/data/$pbf_file" "${REGIONS[$region]}" || {
            echo " Erreur lors du téléchargement de $region"
            exit 1
        }
        echo "   $region téléchargé"
    else
        echo "   $region déjà présent"
    fi
done

# Fusion des fichiers PBF
echo " Fusion des fichiers PBF..."
pbf_files=""
for region in "${!REGIONS[@]}"; do
    pbf_files="$pbf_files /data/${region}-latest.osm.pbf"
done

osmium merge $pbf_files -o /data/combined.osm.pbf
echo "  Fichiers PBF fusionnés"

# Génération des données pour chaque profil
echo "  Génération des données OSRM pour les profils : ${PROFILES[*]}"

for profile in "${PROFILES[@]}"; do
    echo "   Traitement du profil : $profile"
    
    # Extraction OSRM pour le profil
    echo "    Extraction des données pour $profile..."
    osrm-extract -p /opt/$profile.lua /data/combined.osm.pbf
    
    # Partitionnement
    echo "    Partitionnement pour $profile..."
    osrm-partition /data/combined.osrm
    
    # Customisation
    echo "    Customisation pour $profile..."
    osrm-customize /data/combined.osrm
    
    # Renommage du fichier pour le profil
    mv /data/combined.osrm /data/combined-$profile.osrm
    
    echo "    Profil $profile terminé"
done

# Nettoyage des fichiers temporaires
echo " Nettoyage des fichiers temporaires..."
for region in "${!REGIONS[@]}"; do
    rm -f "/data/${region}-latest.osm.pbf"
done
rm -f /data/combined.osm.pbf

echo " Données OSRM prêtes pour les 3 régions !"
echo " Régions couvertes :"
echo "   - Auvergne Rhône-Alpes"
echo "   - Île-de-France" 
echo "   - Bretagne"
echo ""
echo " Profils disponibles :"
for profile in "${PROFILES[@]}"; do
    echo "   - $profile"
done
echo ""
echo "  Démarrage du service OSRM..."
