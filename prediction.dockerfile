#predicter
# Base image
FROM python:3.9-slim

EXPOSE $PORT

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
#Directories
COPY requirements.txt requirements.txt
COPY src/ src/
COPY src/models/deployment.py src/models/deployment.py
#Enviroment

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

CMD exec uvicorn src.models.deployment:app --port $PORT --host 0.0.0.0 --workers 1