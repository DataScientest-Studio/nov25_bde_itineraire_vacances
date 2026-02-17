FROM python:3.11-slim

WORKDIR /app

# dépendances système
RUN apt-get update && apt-get install -y cron

# copier requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copier projet
COPY . /app

# ajouter cron
COPY docker/cron/crontab /etc/cron.d/etl-cron
RUN chmod 0644 /etc/cron.d/etl-cron && crontab /etc/cron.d/etl-cron

CMD ["cron", "-f"]
