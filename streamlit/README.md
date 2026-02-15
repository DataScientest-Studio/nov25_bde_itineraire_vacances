# Streamlit TripMaNGO - Interface Web

Application web Streamlit pour l'interface utilisateur de TripMaNGO, permettant la recherche et la visualisation d'itinéraires de vacances.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit     │ ──▶ │   API FastAPI    │ ──▶│   PostgreSQL    │
│   :8501         │     │   :8000          │     │   :5432         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Folium Maps   │     │   OSRM Routing     │     │   POIs Data     │
│   Visualisation │     │   Services         │     │   Categories    │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
```

---

## Structure du Projet

```
streamlit/
├── main.py                 # Point d'entrée principal
├── pages/                  # Pages de l'application
│   ├── search.py          # Page de recherche d'itinéraires
│   ├── results.py         # Page de résultats et visualisation
│   └── media/             # Images et assets
├── utils/                  # Utilitaires et fonctions partagées
│   ├── utils.py           # Fonctions API et utilitaires
│   └── __init__.py
├── data/                   # Données statiques
├── .streamlit/            # Configuration Streamlit
├── requirements.txt        # Dépendances Python
├── Dockerfile            # Configuration Docker
└── README.md              # Documentation
```

---

## Démarrage Rapide

### 1. Avec Docker (Recommandé)

```bash
# Construction de l'image
docker build -t streamlit-tripmango .

# Lancement du service
docker run -p 8501:8501 streamlit-tripmango

# Ou via docker-compose
docker-compose up streamlit
```

### 2. En Local

```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancement de l'application
streamlit run main.py

# Accès à l'application
# http://localhost:8501
```

---

## Fonctionnalités

### **Page de Recherche**
- **Sélection de commune** : Recherche par nom ou coordonnées
- **Catégories POI** : Filtres multi-catégories avec sous-catégories
- **Paramètres de voyage** :
  - Rayon de recherche (max 30 km)
  - Durée du séjour (max 30 jours)
  - Mode de transport (à pied, Voiture, Vélo)
- **Solver** : Choix de l'algorithme d'optimisation

### **Page de Résultats**
- **Carte interactive** : Visualisation Folium des itinéraires
- **Détails du voyage** : POIs visités, distances, temps
- **Export** : Téléchargement des résultats
- **Navigation** : Retour/modification de la recherche

---

## Configuration

### Variables d'Environnement

```bash
# API TripMaNGO
API_BASE_URL=http://api:8000

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Configuration Docker

```dockerfile
FROM python:3.13-slim
WORKDIR /app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl --fail http://localhost:8501/_stcore/health
```

---

## Dépendances Principales

| Package | Version | Usage |
|---------|---------|-------|
| **streamlit** | 1.54.0 | Framework web principal |
| **folium** | 0.20.0 | Cartes interactives |
| **streamlit-folium** | 0.26.1 | Intégration Folium/Streamlit |
| **pandas** | 2.3.3 | Manipulation de données |
| **requests** | 2.32.5 | Appels API |
| **altair** | 6.0.0 | Visualisations |

---

## Flux Utilisateur

### 1. **Recherche**
```
Utilisateur → Page Recherche → API FastAPI → PostgreSQL
                ↓
         Sélection POI → Construction Payload
```

### 2. **Calcul**
```
Payload → API Compute → OSRM Services → Algorithmes
                ↓
         Résultats Itinéraire → Stockage
```

### 3. **Visualisation**
```
Résultats → Page Results → Folium Maps → Interface Utilisateur
```

---

### Utilitaires Disponibles

```python
# Appels API
from utils.utils import fetch_main_categories, fetch_sub_categories

# Session State
st.session_state.payload
st.session_state.itinerary_payload
```
---
[Retour sur la documentation principale](../README.md)