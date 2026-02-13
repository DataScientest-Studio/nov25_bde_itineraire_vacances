FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY app.py .

# Exposer le port
EXPOSE 8001

# Démarrer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
