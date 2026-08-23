FROM python:3.10-slim

WORKDIR /app

COPY requirements_docker.txt .

RUN pip install --no-cache-dir "setuptools<81" wheel

RUN pip install --no-cache-dir -r requirements_docker.txt

RUN python -m nltk.downloader stopwords wordnet

COPY flask_app/ /app/flask_app/

COPY artifacts/data/vectorized/vectorizer.pkl /app/artifacts/data/vectorized/vectorizer.pkl

EXPOSE 5000

CMD ["python", "flask_app/app.py"]