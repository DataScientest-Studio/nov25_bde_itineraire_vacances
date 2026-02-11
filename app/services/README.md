# Services - Couche Métier de l'API

La couche `services` constitue la logique métier de l'application TripMaNGo. Elle orchestre les interactions entre l'API FastAPI et les pipelines de calcul tout en assurant la transformation des données et la gestion des erreurs.

## Architecture des Services

```
API FastAPI
    ↓
Services (Couche Métier)
    ↓
Pipeline (Calcul Algorithmique)
    ↓
External APIs (OSRM, PostgreSQL)
```

---

## ItineraryService - Service Principal d'Itinéraires

### Rôle et Responsabilités

Le `ItineraryService` est l'orchestrateur principal qui :
1. **Convertit** les POIs de l'API en DataFrame Polars
2. **Déclenche** le pipeline complet de calcul
3. **Enrichit** les résultats avec les métadonnées
4. **Formate** la réponse finale pour l'API
5. **Gère** les erreurs et cas limites

### Signature du Service

```python
class ItineraryService:
    """
    Service orchestrant :
    - conversion des POIs en DataFrame
    - appel du pipeline (clustering, OSRM, solveur, enrichissement)
    - formatage final pour l'API
    """
```

### Initialisation

```python
def __init__(self, osrm_client: OSRMClientAsync):
    self.osrm = osrm_client
    self.pipeline = ItineraryPipeline()
```

**Dépendances** :
- `OSRMClientAsync` : Client pour les calculs de distance
- `ItineraryPipeline` : Pipeline de calcul d'itinéraires

---

## Méthode Principale : `compute_itinerary()`

### Signature Complète

```python
async def compute_itinerary(
    self,
    pois: List[Dict[str, Any]],
    days: int,
    transport_mode: str,
    solver: str,
    start_lat: float,
    start_lon: float,
) -> Dict[str, Any]
```

### Paramètres d'Entrée

| Paramètre | Type | Description |
|-----------|------|-------------|
| `pois` | `List[Dict]` | Liste des POIs pré-filtrés par l'API |
| `days` | `int` | Nombre de jours du voyage |
| `transport_mode` | `str` | Mode de transport (`walk`, `bike`, `car`) |
| `solver` | `str` | Solveur d'optimisation (`nn2o`, `ga`, `auto`) |
| `start_lat` | `float` | Latitude du point de départ |
| `start_lon` | `float` | Longitude du point de départ |

### Flux d'Exécution

#### Étape 0 - Préparation des Métadonnées

```python
# Extraction des métadonnées pour enrichissement final
meta = {
    poi.poi_id: {
        "nom_du_poi": poi.nom_du_poi,
        "description": poi.description,
        "adresse": poi.adresse,
        "contact_phone": poi.contact_phone,
        "contact_mail": poi.contact_mail,
        "contact_website": poi.contact_website,
        "itineraire": poi.itineraire,
        "h3_r7": poi.h3_r7,
        "diversity_commune_norm": poi.diversity_commune_norm,
    }
    for poi in pois
}
```

#### Étape 1 - Conversion DataFrame Polars

```python
pois_df = pl.DataFrame([
    {
        "poi_id": poi.poi_id,
        "nom_du_poi": poi.nom_du_poi,
        "latitude": poi.latitude,
        "longitude": poi.longitude,
        "main_category": poi.main_category,
        "sub_category": poi.sub_category,
        "h3_r7": poi.h3_r7,
        "diversity_commune_norm": poi.diversity_commune_norm,
        "final_score": poi.final_score,
    }
    for poi in pois
])
```

**Colonnes essentielles** :
- **Localisation** : `latitude`, `longitude`
- **Catégorisation** : `main_category`, `sub_category`
- **Scoring** : `final_score`, `diversity_commune_norm`
- **Indexation** : `h3_r7` (géospatial)

#### Étape 2 - Pipeline de Calcul [En savoir plus](../pipeline/README.md)

```python
df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer = (
    await self.pipeline.run_from_pois_df(
        pois_df=pois_df,
        nb_days=days,
        anchor_lat=start_lat,
        anchor_lon=start_lon,
        osrm=self.osrm,             
        transport_mode=transport_mode,
        solver=solver,
    )
)
```

**Déclenchement du pipeline complet** :
- Clustering spatial des POIs
- Calcul des matrices OSRM
- Optimisation par solveur
- Enrichissement des métriques

#### Étape 3 - Gestion des Cas Limites

```python
# Aucun itinéraire trouvé
if df_itinerary.is_empty():
    return {
        "itinerary": [],
        "trip_total_distance_km": 0.0,
        "trip_total_duration_min": 0.0,
        "optimizer": optimizer,
    }
```

#### Étape 4 - Formatage Final pour l'API

##### Construction des POIs Enrichis

```python
poi_payload = {
    # Champs du pipeline
    "osrm_index": row["osrm_index"],
    "cluster_id": row["cluster_id"],
    "poi_id": row["poi_id"],
    "latitude": row["latitude"],
    "longitude": row["longitude"],
    "main_category": row["main_category"],
    "sub_category": row.get("sub_category"),
    "final_score": row["final_score"],
    "order": row["order"],
    "solver_used": row["solver_used"],
    
    # Métriques calculées
    "distance_from_prev_km": row["distance_from_prev_km"],
    "duration_from_prev_min": row["duration_from_prev_min"],
    "cumulative_distance_km": row["cumulative_distance_km"],
    "cumulative_duration_min": row["cumulative_duration_min"],
    "day_total_distance_km": row["day_total_distance_km"],
    "day_total_duration_min": row["day_total_duration_min"],
    
    # Métadonnées originales
    "nom_du_poi": m.get("nom_du_poi"),
    "description": m.get("description"),
    "adresse": m.get("adresse"),
    "contact_phone": m.get("contact_phone"),
    "contact_mail": m.get("contact_mail"),
    "contact_website": m.get("contact_website"),
    "itineraire": m.get("itineraire"),
}
```

##### Géométrie OSRM par Jour

```python
# Récupération de la géométrie complète pour chaque jour
coords_day = [
    (row["longitude"], row["latitude"])
    for row in df_day.to_dicts()
]

osrm_route = await self.osrm.route_full(coords_day, profile=transport_mode)
```

##### Structure de Réponse Finale

```python
result_days.append({
    "day": int(cluster_id),
    "pois": pois_for_day,
    "total_distance_km": day_total_distance_km,
    "total_duration_min": day_total_duration_min,
    "geometry": osrm_route["geometry"],  # GeoJSON pour carte
})
```

#### Étape 5 - Agrégation des Totaux

```python
trip_total_distance = sum(day["total_distance_km"] for day in result_days)
trip_total_duration = sum(day["total_duration_min"] for day in result_days)

return {
    "itinerary": result_days,
    "trip_total_distance_km": trip_total_distance,
    "trip_total_duration_min": trip_total_duration,
    "optimizer": optimizer,
}
```

---

## Structure de Réponse API

### Format Complet

```json
{
  "itinerary": [
    {
      "day": 0,
      "pois": [
        {
          "poi_id": 123,
          "nom_du_poi": "Tour Eiffel",
          "latitude": 48.8584,
          "longitude": 2.2945,
          "main_category": "Patrimoine & Monuments",
          "sub_category": "Monuments",
          "final_score": 0.95,
          "order": 0,
          "solver_used": "nn2o",
          "distance_from_prev_km": 0.0,
          "duration_from_prev_min": 0.0,
          "cumulative_distance_km": 0.0,
          "cumulative_duration_min": 0.0,
          "day_total_distance_km": 2.3,
          "day_total_duration_min": 28,
          "description": "Iconique monument parisien...",
          "adresse": "Champ de Mars, 5 Avenue Anatole France...",
          "contact_phone": "+33 8 92 70 12 39",
          "itineraire": true
        }
      ],
      "total_distance_km": 2.3,
      "total_duration_min": 28,
      "geometry": "encoded_polyline_string"
    }
  ],
  "trip_total_distance_km": 15.2,
  "trip_total_duration_min": 185,
  "optimizer": "auto"
}
```

### Champs par Type

#### Champs Pipeline
- `osrm_index`, `cluster_id`, `order`, `solver_used`
- Métriques de distance/durée

#### Champs POI
- `poi_id`, `nom_du_poi`, `latitude`, `longitude`
- `main_category`, `sub_category`, `final_score`

#### Champs Métadonnées
- `description`, `adresse`, `contact_*`
- Informations pratiques pour l'utilisateur

#### Champs Géométriques
- `geometry` : Polyligne encodée OSRM pour affichage carte

---

## Debug et Monitoring

### Logs Intégrés

```python
def debug_step(self, df, step_name):
    logger.info(f"=== {step_name} ===")
    logger.info(f"Total POIs : {df.shape[0]}")
```

**Points de contrôle** :
1. **Chargement initial** : POIs bruts
2. **Après clustering** : Répartition par jour
3. **Après OSRM distance** : Matrice distances
4. **Après OSRM durée** : Matrice durées
5. **Après solveur** : Itinéraire optimisé
6. **Formatage final** : Résultat prêt pour API

### Exemple de Logs

```
=== 1. Chargement initial ===
Total POIs : 156
=== 2. Après clustering ===
Total POIs : 89
=== 3. Après OSRM distance ===
Total POIs : 89
=== 4. Après OSRM durée ===
Total POIs : 89
=== 5. Après solveur ===
Total POIs : 67
=== 6. Formatage final ===
Total POIs : 67
```

---

## Intégration avec l'API FastAPI

### Injection de Dépendances

```python
# Dans dependencies.py
def get_itinerary_service(
    osrm=Depends(get_osrm_client),
) -> ItineraryService:
    return ItineraryService(osrm_client=osrm)
```

### Utilisation dans l'Endpoint

```python
# Dans itinerary.py
@router.post("/compute")
async def compute_itinerary(
    request: ItineraryRequest,
    service: ItineraryService = Depends(get_itinerary_service),
):
    # Récupération des POIs filtrés
    pois = await get_filtered_pois(request.categories, request.bounds)
    
    # Appel au service
    result = await service.compute_itinerary(
        pois=pois,
        days=request.days,
        transport_mode=request.transport_mode,
        solver=request.solver,
        start_lat=request.latitude,
        start_lon=request.longitude,
    )
    
    return result
```

---

## Gestion des Erreurs

### Cas Limites Gérés

1. **Aucun POI trouvé** : Retour itinéraire vide
2. **Pipeline échoue** : Logging et fallback
3. **OSRM indisponible** : Timeout et retry
4. **Solveur échoue** : Basculement automatique

### Robustesse

```python
# Vérification des résultats
if df_itinerary.is_empty():
    return empty_itinerary_response()

# Validation des coordonnées
if not all(-90 <= lat <= 90 and -180 <= lon <= 180 for lat, lon in coords_day):
    raise ValueError("Coordonnées invalides")
```

---

## Performance et Optimisations

### Optimisations Implémentées

1. **DataFrame Polars** : Performance vs Pandas
2. **Async OSRM** : Requêtes parallèles
3. **Lazy Loading** : Métadonnées chargées à la demande
4. **Memory Efficient** : Streaming des gros datasets

### Métriques de Performance

| Opération | Temps moyen | Optimisation |
|-----------|-------------|--------------|
| Conversion DataFrame | < 10ms | Polars |
| Pipeline complet | 1-5s | Async + caching |
| Formatage API | < 100ms | Vectorisation |
| Géométrie OSRM | 200-500ms | Requête unique/jour |

---

## Tests et Validation (TODO)

### Tests Unitaires
### Tests d'Intégration

---

## Évolutions Possibles

### Améliorations Court Terme

1. **Cache intelligent** : Mémorisation des itinéraires similaires
2. **Validation avancée** : Vérification cohérence géographique
3. **Metrics détaillées** : Temps de traitement par étape

### Évolutions Long Terme

1. **Multi-services** : Découpage en services spécialisés
2. **Event-driven** : Architecture asynchrone complète
3. **ML integration** : Apprentissage des préférences

---

[Retour sur la documentation de l'API](../README.md)