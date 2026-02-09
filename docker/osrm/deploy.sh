#!/bin/bash

# Script de déploiement du service OSRM multi-régions
echo " Déploiement du service OSRM multi-régions..."

# Arrêt des containers existants
echo " Arrêt des containers existants..."
docker-compose down

# Suppression des anciennes données
read -p "Voulez-vous supprimer et retélécharger les données OSRM ? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo " Suppression des anciennes données..."
    docker volume rm osrm_osrm_data 2>/dev/null || true
fi

# Construction et démarrage du service
echo " Construction et démarrage d'OSRM..."
docker-compose up --build -d

echo ""
echo "  Préparation des données OSRM en cours..."
echo "    Auvergne Rhône-Alpes"
echo "    Île-de-France" 
echo "    Bretagne"
echo ""
echo "  Temps estimé : 5-15 minutes"
echo ""

# Surveillance de la progression
echo " Surveillance du démarrage..."
while true; do
    if docker-compose logs osrm-multi-regions 2>&1 | grep -q " Démarrage du service OSRM"; then
        echo " OSRM est prêt !"
        break
    fi
    
    if docker-compose ps osrm-multi-regions | grep -q "unhealthy"; then
        echo " Le service OSRM rencontre des problèmes"
        echo " Logs détaillés :"
        docker-compose logs --tail=20 osrm-multi-regions
        exit 1
    fi
    
    echo " Préparation en cours... ($(date))"
    sleep 30
done

# Test final
echo " Test final du service..."
sleep 5
docker-compose run --rm osrm-test

echo ""
echo " Service OSRM multi-régions déployé avec succès !"
echo ""
echo " Accès au service : http://localhost:5000"
echo " Statut : http://localhost:5000/health"
echo ""
echo " Régions couvertes :"
echo "   - Auvergne Rhône-Alpes"
echo "   - Île-de-France"
echo "   - Bretagne"
echo ""
echo "📝 Exemple d'appel API :"
echo "curl 'http://localhost:5000/route/v1/driving/2.35,48.85;2.36,48.86?overview=false'"
echo ""
echo "Pour voir les logs : docker-compose logs -f"
echo "Pour arrêter : docker-compose down"
