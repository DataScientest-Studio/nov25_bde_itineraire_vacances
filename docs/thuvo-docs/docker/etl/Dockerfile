# C:\Users\DELL\Downloads\ItineraireVacances3\thuvo-docs\docker\etl\Dockerfile.etl

FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_etl.py .

CMD ["python", "run_etl.py"]
