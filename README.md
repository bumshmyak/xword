# XWord

Russian crosswords solver. Takes an image of the crossword (together with questions), recognizes and solves it.

### Install

1. Download [data](https://drive.google.com/uc?export=download&id=135R68nLquBKDQVeaWeIlC2AIMa-CWPWF "data")
2. `pip install -r requirements.txt`
3. [Setup Google Cloud Vision API](https://codelabs.developers.google.com/codelabs/cloud-vision-api-python#4 "Setup Google Cloud Vision API"). Make sure to set GOOGLE_APPLICATION_CREDENTIALS env variable to point to your key.json file.
4. `c++ -O3 solver.cc -o solver`

### Usage

```bash
python main.py {DATA_PATH} {IMG_PATH}
```
