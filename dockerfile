FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir "setuptools<81" wheel

RUN pip install --no-cache-dir -r requirements_docker.txt

RUN python -m nltk.downloader stopwords wordnet

COPY flask_app/ /app/

COPY artifacts/data/vectorized/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5000

CMD ["python", "app.py"]