#the kaggle.json containning a token for kaggle needs to located in the .kaggle folder for the import to work
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import pytest
import time

api = KaggleApi()
api.authenticate()

path = r"./src/data"


#def test_download_files():


def download_data(path):
    try:
        api.competition_download_files("tweet-sentiment-extraction", path = path)
    except Exception:
        print("Didn't download data")




def extract_data(path):
    try:
        with zipfile.ZipFile(path+"/tweet-sentiment-extraction.zip", 'r') as zip_ref:
            zip_ref.extractall(path)
    except FileNotFoundError:
        print("Couldn't extract data from Zip-file")
        assert False


download_data(path)
extract_data(path)





def test_download_files(capsys):
    extract_data(path)
    captured = capsys.readouterr()
    
