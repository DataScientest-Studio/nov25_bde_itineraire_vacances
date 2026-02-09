#!/bin/bash

# Script de déploiement de l'API FastAPI avec OSRM et PostgreSQL
echo "🚀 Déploiement de l'API FastAPI..."

# Création des répertoires nécessaires
mkdir -p ../osrm_data
mkdir -p ../logs

# Arrêt des containers existants
echo "🛑 Arrêt des containers existants..."
docker-compose down

# Construction et démarrage des services
echo "🔧 Construction et démarrage des services..."
docker-compose up --build -d

# Attente du démarrage des services
echo "⏳ Attente du démarrage des services..."
sleep 15

# Vérification du statut
echo "📊 Vérification du statut des services..."
docker-compose ps

echo ""
echo "✅ API disponible !"
echo "🔗 API FastAPI: http://localhost:8000"
echo "📚 Documentation: http://localhost:8000/docs"
echo "🗺️  OSRM Backend: http://localhost:5000"
echo "🗄️  PostgreSQL: localhost:5433"
echo ""
echo "Pour voir les logs: docker-compose logs -f"
echo "Pour arrêter: docker-compose down"
