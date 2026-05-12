# StaySure AI

Smart room rent prediction using room images and metadata.

This repository is being built in phases. The first completed slice is the Kaggle-ready model training pipeline in `Model/`.

## Repository Structure

```text
Backend/      FastAPI backend source, added in the API phase
Dataset/      Local dataset notes and optional room image storage
Docs/         API, deployment, and product documentation
Frontend/     React frontend source, added in the UI phase
Model/        Training, inference, preprocessing, and similarity code
PLAN.md       Full project execution plan
```

## Train the Model in Kaggle

Upload your room images as a Kaggle dataset and keep the CSV columns compatible with:

```text
id,img1,img2,img3,img4,img5,location,size_sqft,furnishing_type,cleanliness_score,amenities,rent_price,bhk
```

If your Google Sheet is public, the training script can download it directly:

```bash
python Model/train.py \
  --sheet-url "https://docs.google.com/spreadsheets/d/1kEFA6MHvewYsFtcAAEntbBfZc-5yZeUXf1mtM47QlTE/edit?gid=0#gid=0" \
  --image-root "/kaggle/input/staysure-room-images/rooms" \
  --output-dir "/kaggle/working/staysure_artifacts" \
  --epochs 25 \
  --batch-size 16 \
  --folds 5
```

If the sheet is private, export it as CSV and run:

```bash
python Model/train.py \
  --csv-path "/kaggle/input/staysure-dataset/rooms.csv" \
  --image-root "/kaggle/input/staysure-room-images/rooms" \
  --output-dir "/kaggle/working/staysure_artifacts" \
  --epochs 25 \
  --batch-size 16 \
  --folds 5
```

Generated files:

```text
best_model.keras
final_model.keras
preprocessor.joblib
training_report.json
similar_rooms_index.csv
similar_rooms_embeddings.npz
```

## Test Inference After Training

```bash
python Model/predict.py \
  --model "/kaggle/working/staysure_artifacts/final_model.keras" \
  --preprocessor "/kaggle/working/staysure_artifacts/preprocessor.joblib" \
  --similar-index "/kaggle/working/staysure_artifacts/similar_rooms_index.csv" \
  --similar-embeddings "/kaggle/working/staysure_artifacts/similar_rooms_embeddings.npz" \
  --images "/kaggle/input/sample-room/img1.jpg" "/kaggle/input/sample-room/img2.jpg" \
  --metadata '{"location":"Delhi","bhk":"1BHK","size_sqft":250,"furnishing_type":"Semi-Furnished","cleanliness_score":8,"amenities":"wifi,ac,attached bathroom"}'
```

## Local Python Setup

```bash
cd Model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker Model Training

Place `rooms.csv` in `Dataset/` and room images in `Dataset/rooms/`, then run:

```bash
docker compose run --rm model-trainer
```
