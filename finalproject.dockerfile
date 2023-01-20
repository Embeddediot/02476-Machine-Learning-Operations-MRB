# Base image
FROM python:3.8.13
#Directories
COPY requirements.txt requirements.txt
COPY src/ src/
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
ENTRYPOINT ["python", "-m", "src.models.deployment"]
