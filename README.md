# creative-abs-summ-sketch

- Create folder `SUMMscreen/fd/` and download the data from the GoogleDrive: https://drive.google.com/drive/folders/1IQwlkIoIBFFUD-dwmQAUTdOjD5tBMXKs?usp=sharing
- Install `requirements.txt` in a virtual environment and activate the environment.

## Training
```
run_train.sh
```
Batch size 2 on 48GB VRAM.
Suitable data: `clean_data/train_plus_valid.json` from the Google Drive.
Make sure to adjust the paths.

## Evaluation 
```
evaluate.sh
```
e.g. for `clean_data/test.json` from the Google Drive.
Make sure to adjust the paths.

## Prediction
(without labels),  e.g. for `clean_data/test_challenge.json` from the Google Drive.
```
predict.sh
```
Make sure to adjust the paths.

## Analysis and format adjusting for the datasets:
See `Exploratory_Analysis.ipynb` notebook.
