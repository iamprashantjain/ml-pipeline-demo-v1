# FROM python:3.10-slim

# WORKDIR /app

# COPY requirements_docker.txt .

# RUN pip install --no-cache-dir "setuptools<81" wheel

# RUN pip install --no-cache-dir -r requirements_docker.txt

# RUN python -m nltk.downloader stopwords wordnet

# COPY flask_app/ /app/flask_app/

# COPY artifacts/data/vectorized/vectorizer.pkl /app/artifacts/data/vectorized/vectorizer.pkl

# EXPOSE 5000

# CMD ["python", "flask_app/app.py"]



# ======================

FROM python:3.10-slim

# Minimize layers by combining RUN commands
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements_docker.txt .

# Combine all pip installations and cleanup in one layer
RUN pip install --no-cache-dir "setuptools<81" wheel && \
    pip install --no-cache-dir -r requirements_docker.txt && \
    python -m nltk.downloader stopwords wordnet && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy application files
COPY flask_app/ /app/flask_app/
COPY artifacts/data/vectorized/vectorizer.pkl /app/artifacts/data/vectorized/vectorizer.pkl

EXPOSE 5000

CMD ["python", "flask_app/app.py"]