# Documentation & Guide d'Utilisation : Pipeline ETL

Ce document détaille le fonctionnement, l'architecture et la procédure d'exécution du pipeline d'ingestion de données (Extract-Transform-Load) du projet Itinéraire Vacances.
1. Architecture du Pipeline

Le pipeline est orchestré par Apache Airflow et s'exécute mensuellement (@monthly).
Étape	Script	Description Technique
1. Extract	1_extract.py	Téléchargement en Streaming du flux GZIP DataTourisme (>2Go) pour économiser la RAM.
2. Transform	2_transform.py	Lecture directe dans le ZIP. Extraction récursive des adresses JSON. Calcul des index géospatiaux H3. Catégorisation par mots-clés.
3. Load	3_load.py	Full Refresh (Truncate). Insertion relationnelle stricte (Adresse → ID → POI) dans PostgreSQL.
2. Guide d'Exécution Pas-à-Pas

Voici la procédure pour lancer le pipeline sur votre machine locale.
📋 Prérequis

    Avoir Docker Desktop installé et lancé.

    Avoir cloné ce dépôt Git.

    Le Webhook Slack est configuré et opérationnel.

🚀 Démarrage de l'Infrastructure

Ouvrez un terminal à la racine du projet et lancez la commande suivante :
Créer un réseau network externe:
```bash
docker network create vacances_network
```

```bash

docker-compose up -d
```
    ⏳ Note : Attendez environ 30 à 60 secondes que les services s'initialisent complètement.

🖥️ Accès à l'Interface Airflow

Pour piloter les scripts :

    URL : http://localhost:8080

    Utilisateur : airflow

    Mot de passe : airflow

🐘 Accès à pgAdmin (Base de Données)

Pour visualiser les tables et les données :

    URL : http://localhost:5050

    Login Interface : admin@admin.com

    Mot de passe : root

    Connecter le Serveur :

        Clic droit sur Servers > Register > Server...

        Onglet General : Nom = Projet Vacances

        Onglet Connection :

            Host name/address : postgres-vacances (⚠️ Important : ne mettez pas "localhost")

            Port : 5432

            Maintenance database : vacances

            Username : vacances_user

            Password : vacances_password

        Cliquez sur Save.

Lancer le script sql pour itiniliser la base de données POSTGRES:

```bash
docker exec -i postgres-vacances psql -U vacances_user -d vacances < ./sql/init_db.sql
```



▶️ Lancer le Pipeline ETL

    Dans Airflow, repérez la ligne etl_vacances_final.

    Activez le DAG en cliquant sur l'interrupteur (Toggle) à gauche (Il doit passer de OFF à ON / Bleu).

    Lancez l'exécution manuelle :

        Cliquez sur le bouton Play ▶️ (sous la colonne Actions à droite).

        Sélectionnez Trigger DAG.

📊 Suivre l'Exécution

Cliquez sur le nom du DAG (etl_vacances_final) puis allez dans l'onglet Grid ou Graph. Le pipeline va exécuter les tâches séquentiellement :

    🟦 extract_data : Téléchargement du flux (~30s).

    🟦 transform_data : Nettoyage & Calculs H3 (~1-2 min).

    🟦 load_data : Insertion en base (~1 min).

    Succès : Les carrés deviennent Verts 🟩.

    Échec : Un carré devient Rouge 🟥.

3. Monitoring & Alertes (Slack)

Le système est connecté à un Webhook Slack et est opérationnel.

    🔴 Alerte Rouge (Failure) : Une notification est envoyée immédiatement sur Slack si une tâche Python échoue (crash, erreur réseau, bug).

    🟢 Notification Verte (Success) : Une confirmation est envoyée quand les 3 étapes (Extract, Transform, Load) sont terminées et que la base est à jour.

4. Schéma de Données (Sortie SQL)

Le pipeline alimente la table poi dans PostgreSQL avec la structure suivante :
Colonne	Type	Description
poi_id	SERIAL	Clé primaire unique.
nom_du_poi	VARCHAR	Nom du lieu touristique.
latitude / longitude	FLOAT	Coordonnées GPS.
h3_r9	VARCHAR	Index spatial H3 (résolution 9) pour recherche de proximité.
main_category_id	INT	Clé étrangère vers la table main_category.
sub_category_id	INT	Clé étrangère vers la table sub_category.
final_score	FLOAT	Score normalisé (0-1) pour le classement.
5. Commandes Utiles (Debug)

Vérifier le nombre de lieux en base (via Terminal) :
```bash

docker exec -it postgres-vacances psql -U vacances_user -d vacances -c "SELECT count(*) FROM poi;"
```

Redémarrer proprement l'infrastructure :
```bash

docker-compose down && docker-compose up -d
```

[Retour sur la documentation principale](../README.md)