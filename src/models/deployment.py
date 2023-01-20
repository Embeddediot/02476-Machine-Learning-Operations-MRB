from typing import Union
from src.models.model import TwitterSentimentAnalysis
from scipy.special import softmax
import numpy as np

from fastapi import FastAPI

app = FastAPI()

model = TwitterSentimentAnalysis()

@app.get("/")
def read_root():
    return {"Input": "Tweet"}


@app.get("/tweets/{tweet}")
def read_item(tweet: str):
    encoded_input = model.tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    prediction = ""
    for i in range(scores.shape[0]):
        l = model.config.id2label[ranking[i]]
        s = scores[ranking[i]]
        prediction += f"{i+1}) {l} {np.round(float(s), 4)} "

    return f'{prediction}'

