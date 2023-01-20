FROM krishansubudhi/transformers_pytorch
FROM redislabs/rcp-gcloud
#RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y
COPY . /02476-Machine-Learning-Operations-MRB
WORKDIR /02476-Machine-Learning-Operations-MRB
CMD python3 src/data/make_dataset.py