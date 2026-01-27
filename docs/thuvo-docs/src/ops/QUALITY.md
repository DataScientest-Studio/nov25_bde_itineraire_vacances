# QUALITY & OPS — Itinéraire Vacances (PRIME)

## Pourquoi cette démarche ?
Ce projet vise à démontrer le passage d’un prototype analytique
à un **système data gouverné**, capable de produire des résultats
fiables, traçables et reproductibles.

---

## 1. Observabilité
Tous les composants (ETL, API, scoring PRIME) produisent des logs JSON
structurés et homogènes.

**Objectifs :**
- comprendre ce qui s’est passé
- mesurer les performances
- diagnostiquer rapidement une anomalie

**Exemples d’événements suivis :**
- durée des étapes ETL
- nombre de POI ingérés / filtrés
- temps de réponse API
- distribution des scores PRIME

---

## 2. Qualité des données PRIME (Gold)
Avant toute exposition à l’utilisateur, les résultats PRIME sont validés.

### Contrôles critiques
1. **Schéma requis** : Garantit la présence des champs indispensables au scoring.
2. **Unicité de la clé métier (`poi_id`)** : Évite les doublons et biais de ranking.
3. **Types et bornes** : Détecte les valeurs incohérentes (coordonnées, poids, scores).
4. **Formule de score** : Assure la conformité entre la logique documentée et les résultats.
5. **Anomalies de distribution** : Détecte des scores aberrants révélateurs d’un bug ou d’une dérive.

**Politique :**
- ERROR → blocage ou fallback sécurisé
- WARN → résultat autorisé mais tracé et monitoré

---

## 3. Tests
- Tests techniques : API, paramètres invalides, services externes
- Tests métier : cohérence du ranking selon format / tempo
- Tests de non-régression sur zones de référence

---

## 4. Reproductibilité
- Environnement Docker (DB, ETL, API)
- Configuration par variables d’environnement
- Scripts simples pour lancer une démo complète

---

## Objectif final
Assurer un **comportement stable**, **explicable** et **démontrable**
du moteur PRIME, y compris en conditions de démo.
