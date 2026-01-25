# Airbnb (Paris) – Données d’hébergement (usage analytique)

## Présentation
Airbnb est utilisé dans le projet **Itinéraire Vacances – Prime** comme **source analytique indirecte** pour caractériser l’offre d’hébergement sur un territoire (Paris).

Les données Airbnb ne sont pas utilisées pour afficher ou promouvoir des annonces spécifiques.

---

## Nature des données utilisées

Les données exploitées sont limitées à :
- typologie d’hébergement (logement entier, chambre, etc.)
- distribution spatiale des hébergements (latitude, longtitude)
- indicateurs agrégés (prix, max_people)

Les données sont :
- agrégées
- anonymisées
- décorrélées de toute annonce individuelle

---

## Données volontairement exclues

Pour garantir la conformité :
- aucun identifiant Airbnb public
- aucun lien vers des annonces
- aucun contenu descriptif ou image
- aucune information propriétaire redistribuée

Les données ne sont jamais exposées à l’utilisateur final.

---

## Mode d’intégration dans Prime

Airbnb est utilisé pour :
- identifier les **zones d’ancrage potentielles** (v1)
- estimer la capacité d’accueil touristique (v1)
- ajuster les recommandations selon l’offre locale (v2)

Exemples :
- zone avec forte densité d’hébergements → point d’ancrage probable
- zone sous-équipée → recommandations à la journée

---

## Rôle dans le moteur Prime

Les données Airbnb permettent :
- d’améliorer le réalisme des itinéraires
- de mieux calibrer les distances et durées
- de contextualiser les propositions Prime

Airbnb n’est jamais utilisé comme source de POI touristique.

---

## Limites et précautions
- données non exhaustives
- biais urbain / touristique (Paris)
- source propriétaire

Ces limites sont compensées par :
- l’utilisation prioritaire de DataTourisme
- des règles de fallback basées sur la géographie
