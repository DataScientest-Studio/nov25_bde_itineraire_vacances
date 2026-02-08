#!/bin/bash

# Script d'exemples d'utilisation des différents profils OSRM

echo "  Exemples d'utilisation des profils OSRM"
echo "  ======================================"
echo ""

# Test du profil car (véhicule)
echo "  1. Profil CAR (véhicule) :"
echo "     Route en voiture entre Paris et Lyon :"
echo "     curl 'http://localhost:5000/route/v1/driving/2.35,48.85;4.85,45.75?overview=false'"
echo ""

echo "     Distance et temps en voiture :"
echo "     curl 'http://localhost:5000/route/v1/driving/2.35,48.85;4.85,45.75?overview=full&geometries=geojson'"
echo ""

# Test du profil bike (vélo)
echo "  2. Profil BIKE (vélo) :"
echo "     Route en vélo dans Paris :"
echo "     curl 'http://localhost:5002/route/v1/cycling/2.35,48.85;2.37,48.86?overview=false'"
echo ""

echo "     Pistes cyclables entre deux points :"
echo "     curl 'http://localhost:5002/route/v1/cycling/2.29,48.85;2.37,48.86'"
echo ""

# Test du profil walk (piéton)
echo "  3. Profil WALK (piéton) :"
echo "     Route piétonne dans le centre de Paris :"
echo "     curl 'http://localhost:5001/route/v1/walking/2.34,48.85;2.35,48.86?overview=false'"
echo ""

echo "     Itinéraire piéton avec détails :"
echo "     curl 'http://localhost:5001/route/v1/walking/2.34,48.85;2.35,48.86?steps=true'"
echo ""

# Exemples de matrices
echo "  4. Matrices de distances par profil :"
echo "     Matrice voiture (car) :"
echo "     curl 'http://localhost:5000/table/v1/driving/2.35,48.85;2.37,48.86;2.39,48.87'"
echo ""

echo "     Matrice vélo (bike) :"
echo "     curl 'http://localhost:5002/table/v1/cycling/2.35,48.85;2.37,48.86;2.39,48.87'"
echo ""

echo "     Matrice piéton (walk) :"
echo "     curl 'http://localhost:5001/table/v1/walking/2.35,48.85;2.37,48.86;2.39,48.87'"
echo ""

# Changement de profil
echo "  5. Changer de profil :"
echo "     Pour utiliser le profil vélo :"
echo "     docker-compose down"
echo "     OSRM_PROFILE=bike docker-compose up -d"
echo ""

echo "     Pour utiliser le profil piéton :"
echo "     docker-compose down"
echo "     OSRM_PROFILE=walk docker-compose up -d"
echo ""

echo "     Pour revenir au profil voiture :"
echo "     docker-compose down"
echo "     OSRM_PROFILE=car docker-compose up -d"
echo ""

# Test de disponibilité
echo "  6. Vérifier le profil actuel :"
echo "     curl -s http://localhost:5000/health | jq '.'"
echo ""

echo "  7. Tester une route dans chaque région :"
echo "     Auvergne Rhône-Alpes (Lyon) :"
echo "     curl 'http://localhost:5000/route/v1/car/4.85,45.75;4.86,45.76'"
echo ""

echo "     Île-de-France (Paris) :"
echo "     curl 'http://localhost:5000/route/v1/car/2.35,48.85;2.36,48.86'"
echo ""

echo "     Bretagne (Rennes) :"
echo "     curl 'http://localhost:5000/route/v1/car/-1.68,48.11;-1.67,48.12'"
echo ""
