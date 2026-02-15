# Configuration OSRM Multi-Régions France

Ce projet configure automatiquement OSRM pour 3 régions françaises (Auvergne-Rhône-Alpes, Bretagne, Île-de-France) avec 3 profils de transport (foot, bike, car).

## Prérequis

- Docker installé
- Docker Compose installé (optionnel mais recommandé)
- Au moins 10 Go d'espace disque libre
- Connexion Internet pour télécharger les données OSM

## Démarrage rapide

Recommandation:
Attention si vous êtes utilisteur Windows, OSRM n'est pa conçu pour Windows natif.
Les outils osrm-extract, osrm-partition, osrm-customize et les profils Lua ne fonctionnent correctement que sous Linux.
<br>Utiliser WSL2 (Ubuntu) comme environnement officiel:

- OSRM nécessite un environnement Linux (filesystem, symlinks, toolchain).
- Les scripts bash ne fonctionnent pas sous PowerShell.
- Les images Docker OSRM utilisent des chemins Linux.
- Les profils Lua doivent être exécutés dans un environnement POSIX.

1. Installer WSL2 :
   
   ```bash
   wsl --install -d Ubuntu
   ```

2. Installer Docker Desktop :
   - Activer "Use the WSL2 backend"
   - Désactiver "Expose daemon on tcp://..." (sécurité)

3. Vérifier que Docker fonctionne dans WSL :
   ```bash
   docker info
   ```


### Option 1: Script automatique (recommandé)

Se mettre dans le dossier docker/osrm/scripts/:

```bash
# Rendre le script exécutable
chmod +x setup-osrm.sh

# Lancer le script
./setup-osrm.sh
```

Le script va automatiquement:
1.  Télécharger les 3 régions depuis Geofabrik
2.  Installer osmium-tool
3.  Merger les fichiers .osm.pbf
4.  Extraire et préparer les données pour chaque profil (foot, bike, car)

### Option 2: Docker Compose (pour démarrer les serveurs)

Une fois le script terminé, démarrez tous les serveurs OSRM:

```bash
docker-compose up -d
```

Cela démarre 3 serveurs OSRM:
- **Foot**: http://localhost:5001
- **Bike**: http://localhost:5002
- **Car**: http://localhost:5000

## Structure des fichiers

```
osrm-data/
├── raw/                                    # Fichiers OSM bruts
│   ├── auvergne-latest.osm.pbf
│   ├── rhone-alpes-latest.osm.pbf
│   ├── bretagne-latest.osm.pbf
│   └── ile-de-france-latest.osm.pbf
├── merged/                                 # Fichier mergé
│   └── france-regions-merged.osm.pbf
└── profiles/                               # Données par profil
    ├── foot/
    │   ├── france-merged.osrm
    │   ├── france-merged.osrm.cells
    │   └── france-merged.osrm.hsgr
    ├── bike/
    │   └── ...
    └── car/
        └── ...
```

## Tests des serveurs

### Test du profil FOOT
```bash
curl "http://localhost:5001/route/v1/foot/4.869949,45.726411;4.8357,45.7640?overview=false"
```

### Test du profil BIKE
```bash
curl "http://localhost:5002/route/v1/bike/4.869949,45.726411;4.8357,45.7640?overview=false"
```

### Test du profil CAR
```bash
curl "http://localhost:5000/route/v1/car/4.380427,45.437685;4.8357,45.7640?overview=false"
```

Ces exemples calculent un itinéraire de Nord de lyon (45.437685, 4.380427) à Lyon (45.7640, 4.8357).

## Exemples d'utilisation

### Calcul d'itinéraire avec géométrie complète
```bash
curl "http://localhost:5000/route/v1/car/4.380427,45.437685;4.8357,45.7640?steps=true&geometries=geojson"
```

### Calcul de matrice de distances
```bash
curl "http://localhost:5000/table/v1/car/4.380427,45.437685;4.8357,45.7640;-1.6778,48.1173"
```

### Recherche du point le plus proche
```bash
curl "http://localhost:5000/nearest/v1/car/4.380427,45.437685"
```

## Commandes utiles

### Voir les logs d'un serveur
```bash
docker logs osrm-foot
docker logs osrm-bike
docker logs osrm-car
```

### Arrêter tous les serveurs
```bash
docker-compose down
```

### Redémarrer un serveur spécifique
```bash
docker restart osrm-foot
```

### Supprimer et recommencer
```bash
docker-compose down
rm -rf osrm-data/
./setup-osrm.sh
docker-compose up -d
```

## Démarrage manuel des serveurs (sans docker-compose)

### Serveur FOOT
```bash
docker run -d --name osrm-foot -p 5001:5000 \
  -v "$(pwd)/osrm-data/profiles/foot:/data" \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld /data/france-merged.osrm
```

### Serveur BIKE
```bash
docker run -d --name osrm-bike -p 5002:5000 \
  -v "$(pwd)/osrm-data/profiles/bike:/data" \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld /data/france-merged.osrm
```

### Serveur CAR
```bash
docker run -d --name osrm-car -p 5000:5000 \
  -v "$(pwd)/osrm-data/profiles/car:/data" \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld /data/france-merged.osrm
```

## Personnalisation

### Ajouter d'autres régions

Modifiez le tableau `REGIONS` dans `setup-osrm.sh`:

```bash
REGIONS=(
    "auvergne-rhone-alpes:https://download.geofabrik.de/europe/france/auvergne-rhone-alpes-latest.osm.pbf"
    "bretagne:https://download.geofabrik.de/europe/france/bretagne-latest.osm.pbf"
    "ile-de-france:https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf"
    "provence:https://download.geofabrik.de/europe/france/provence-alpes-cote-d-azur-latest.osm.pbf"
)
```

### Changer les profils

Les profils disponibles dans OSRM sont:
- `foot` - Marche à pied
- `bike` - Vélo
- `car` - Voiture

## Performances

Le temps de traitement dépend de:
- Taille des régions (Île-de-France + Bretagne + Auvergne + Rhone Alpes ≈ 1-2 Go)
- Puissance de votre machine
- Profil (car est généralement plus rapide que foot)

**Estimations:**
- Téléchargement: 5-15 minutes
- Merge: 1-3 minutes
- Extraction par profil: 10-30 minutes
- **Total: ~1-2 heures**

## Dépannage

### Erreur "osmium: command not found"
Le script installe automatiquement osmium. Si erreur, installez manuellement:
```bash
sudo apt-get update
sudo apt-get install osmium-tool
```

### Serveur ne démarre pas
Vérifiez que les fichiers .osrm existent:
```bash
ls -lh osrm-data/profiles/*/france-merged.osrm*
```

### Port déjà utilisé
Changez les ports dans `docker-compose.yml` ou arrêtez le service qui utilise le port.

## Ressources

- [Documentation OSRM](http://project-osrm.org/)
- [API OSRM](https://github.com/Project-OSRM/osrm-backend/blob/master/docs/http.md)
- [Geofabrik Downloads](https://download.geofabrik.de/)
- [Osmium Tool](https://osmcode.org/osmium-tool/)

## Licence

Ce script est fourni tel quel. Les données OSM sont sous licence ODbL.

---
[Retour sur la documentation principale](../README.md)