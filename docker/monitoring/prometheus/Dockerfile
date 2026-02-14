FROM prom/prometheus:latest

# Copier la configuration personnalisée
COPY prometheus.yml /etc/prometheus/
COPY alert_rules.yml /etc/prometheus/

# Arguments de démarrage
CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.console.libraries=/etc/prometheus/console_libraries", \
     "--web.console.templates=/etc/prometheus/consoles", \
     "--storage.tsdb.retention.time=30d", \
     "--web.enable-lifecycle"]
