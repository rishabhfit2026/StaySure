# Kaggle Training Guide

## 1. Add Inputs

Create or attach two Kaggle inputs:

- CSV dataset containing the StaySure metadata columns.
- Room image dataset containing images referenced by `img1` to `img5`.

If the Google Sheet is public, you can skip uploading the CSV and use `--sheet-url`.

## 2. Install Requirements

Kaggle usually includes TensorFlow, NumPy, Pandas, and scikit-learn. If a package is missing, run this in a notebook cell:

```bash
pip install -r /kaggle/working/StaySure/Model/requirements.txt
```

## 3. Train From Google Sheet

```bash
python /kaggle/working/StaySure/Model/train.py \
  --sheet-url "https://docs.google.com/spreadsheets/d/1kEFA6MHvewYsFtcAAEntbBfZc-5yZeUXf1mtM47QlTE/edit?gid=0#gid=0" \
  --image-root "/kaggle/input/staysure-room-images/rooms" \
  --output-dir "/kaggle/working/staysure_artifacts" \
  --epochs 25 \
  --batch-size 16 \
  --folds 5
```

## 4. Train From Uploaded CSV

```bash
python /kaggle/working/StaySure/Model/train.py \
  --csv-path "/kaggle/input/staysure-dataset/rooms.csv" \
  --image-root "/kaggle/input/staysure-room-images/rooms" \
  --output-dir "/kaggle/working/staysure_artifacts" \
  --epochs 25 \
  --batch-size 16 \
  --folds 5
```

## 5. Expected Outputs

```text
/kaggle/working/staysure_artifacts/best_model.keras
/kaggle/working/staysure_artifacts/final_model.keras
/kaggle/working/staysure_artifacts/preprocessor.joblib
/kaggle/working/staysure_artifacts/training_report.json
/kaggle/working/staysure_artifacts/similar_rooms_index.csv
/kaggle/working/staysure_artifacts/similar_rooms_embeddings.npz
```

## 6. Test One Prediction

```bash
python /kaggle/working/StaySure/Model/predict.py \
  --model "/kaggle/working/staysure_artifacts/final_model.keras" \
  --preprocessor "/kaggle/working/staysure_artifacts/preprocessor.joblib" \
  --similar-index "/kaggle/working/staysure_artifacts/similar_rooms_index.csv" \
  --similar-embeddings "/kaggle/working/staysure_artifacts/similar_rooms_embeddings.npz" \
  --images "/kaggle/input/sample-room/img1.jpg" \
  --metadata '{"location":"Delhi","bhk":"1BHK","size_sqft":250,"furnishing_type":"Semi-Furnished","cleanliness_score":8,"amenities":"wifi,ac,attached bathroom"}'
```

