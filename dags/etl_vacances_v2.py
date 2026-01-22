from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import requests 
import json
import logging

# CONFIGURATION SLACK
import os

# Airflow va chercher la variable définie dans le docker-compose
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def send_slack_alert(message_text, color="#FF0000"):
    """Fonction générique pour envoyer un message à Slack"""
    payload = {
        "text": message_text, # Texte de secours pour les notifs mobiles
        "attachments": [
            {
                "color": color, # Rouge pour erreur, Vert pour succès
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message_text
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code != 200:
            logging.error(f"Erreur Slack : {response.status_code} - {response.text}")
        else:
            logging.info("✅ Notification Slack envoyée avec succès !")
    except Exception as e:
        logging.error(f"❌ Impossible de contacter Slack : {e}")

# --- CALLBACKS (Les déclencheurs) ---

def on_failure_callback(context):
    """Se lance si une tâche PLANTE"""
    dag_id = context.get('task_instance').dag_id
    task_id = context.get('task_instance').task_id
    execution_date = context.get('execution_date')
    
    message = (
        f"🔴 *ALERTE CRITIQUE : Échec ETL*\n"
        f"----------------------------------\n"
        f"📍 *DAG* : `{dag_id}`\n"
        f"💀 *Tâche* : `{task_id}`\n"
        f"📅 *Date* : `{execution_date}`\n"
        f"----------------------------------\n"
        f"⚠️ _Intervention requise immédiatement !_"
    )
    send_slack_alert(message, color="#FF0000") # Rouge

def on_success_callback(context):
    """Se lance quand le DAG est FINI avec succès"""
    dag_id = context.get('task_instance').dag_id
    
    message = (
        f"🟢 *SUCCÈS : ETL Terminé*\n"
        f"----------------------------------\n"
        f"🚀 Le pipeline `{dag_id}` a tourné sans erreur.\n"
        f"💾 Les données DataTourisme sont à jour en base.\n"
        f"----------------------------------"
    )
    send_slack_alert(message, color="#36a64f") # Vert

# --- DÉFINITION DU DAG ---

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 0,
    # Si une tâche échoue, on appelle la fonction Slack Failure
    'on_failure_callback': on_failure_callback, 
}

with DAG(
    'etl_vacances_final',
    default_args=default_args,
    description='Pipeline complet DataTourisme avec Alerting Slack',
    schedule_interval='@monthly', 
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['datatourisme', 'production'],
    # Si tout le DAG réussit, on appelle la fonction Slack Success
    on_success_callback=on_success_callback 
) as dag:

    # 1. Extraction
    t1 = BashOperator(
        task_id='extract_data',
        bash_command='python /opt/airflow/src/etl/1_extract.py'
    )

    # 2. Transformation
    t2 = BashOperator(
        task_id='transform_data',
        bash_command='python /opt/airflow/src/etl/2_transform.py'
    )

    # 3. Chargement
    t3 = BashOperator(
        task_id='load_data',
        bash_command='python /opt/airflow/src/etl/3_load.py'
    )

    t1 >> t2 >> t3