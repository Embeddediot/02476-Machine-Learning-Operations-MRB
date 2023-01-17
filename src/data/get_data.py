import zipfile


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

path = r"./src/data"


try:
    api.competition_download_files("tweet-sentiment-extraction", path = path)
except Exception:
    print("Didn't download data")

try:
    with zipfile.ZipFile(path+"/tweet-sentiment-extraction.zip", 'r') as zip_ref:
        zip_ref.extractall(path)
except Exception:
    print("Couldn't extract data from Zip-file")

