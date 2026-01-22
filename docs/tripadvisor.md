# TripAdvisor – Données de popularité et de contexte

## Présentation
TripAdvisor est utilisé dans le projet **Itinéraire Vacances – Prime** comme **source indirecte de signaux de popularité et de contexte**, et non comme base de données de lieux à redistribuer.

Les données issues de TripAdvisor servent exclusivement à :
- enrichir la priorisation des POI (restaurant)
- améliorer la pertinence des recommandations (note moyenne)
- fournir des signaux de fréquentation relative (nombre d'avis)

---

## Nature des données utilisées

Les informations exploitées sont limitées à :
- scores agrégés (ex : note moyenne)
- volumes relatifs (ex : nombre d’avis)
- catégorie générale (restaurant)

Les données **ne sont pas utilisées comme source primaire de POI**.

---

## Données volontairement exclues

Afin de respecter les conditions d’utilisation :
- aucun nom de lieu TripAdvisor n’est affiché
- aucune URL TripAdvisor n’est exposée
- aucun contenu textuel d’avis n’est stocké
- aucune redistribution des données brutes

Les identifiants internes sontremplacés par des identifiants techniques **préfix R**

---

## Mode d’intégration dans Prime

TripAdvisor intervient comme :
- un **facteur de pondération**
- un **signal secondaire** dans le scoring Prime

Exemple conceptuel :
- POI A et POI B similaires
- POI A bénéficie d’un meilleur signal de popularité
- Prime priorise POI A dans l’itinéraire proposé

---

## Rôle dans le moteur Prime
Les données TripAdvisor permettent :
- d’éviter les recommandations trop confidentielles
- d’améliorer la satisfaction utilisateur
- d’équilibrer découverte et attractivité

TripAdvisor n’est jamais utilisé comme source d’autorité géographique.

---

## Limites et précautions
- biais de popularité (zones touristiques très notées)
- couverture inégale selon les territoires (Paris, Annecy)
- dépendance à une source propriétaire

Ces limites sont compensées par :
- la source officielle DataTourisme
- des règles de diversification Prime
