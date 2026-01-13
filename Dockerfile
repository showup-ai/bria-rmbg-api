# Dockerfile pour BRIA-RMBG Server
FROM python:3.11-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY bria_rmbg_server.py .

# Exposer le port
EXPOSE 8080

# Variables d'environnement
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Commande de démarrage avec gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "--workers", "1", "bria_rmbg_server:app"]
