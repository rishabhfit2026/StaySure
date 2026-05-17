# StaySure AI Handoff Plan

This document is written so another LLM or developer can quickly understand what we are building, where the project currently stands, and what should happen next.

## Project Aim

StaySure AI is a rent prediction platform for rooms, PGs, flats, hostels, and similar rental properties.

The core idea is:

- Take up to 5 room/property images.
- Take metadata such as location, BHK, size, furnishing type, cleanliness score, attached bathroom, amenities, and description.
- Predict the expected monthly rent.
- Show similar rooms from the dataset.
- Later connect the model to a backend API and frontend web app.

The main technical goal right now is to finish and validate the machine learning pipeline before building the full application around it.

## Current Repository State

The repository is organized like this:

```text
Backend/      Future FastAPI backend
Dataset/      Dataset collection scripts, CSV, and local room images
Docs/         API and product docs
Frontend/     Future React frontend
Model/        Model training, inference, preprocessing, metrics, and similarity code
PLAN.md       Original phase-based project plan
plan.md       This LLM handoff summary
```

The most important current folder is `Model/`.

Important model files:

```text
Model/train.py                      Main training script
Model/predict.py                    CLI prediction script
Model/KAGGLE.md                     Kaggle training guide
Model/requirements.txt              Python dependencies
Model/src/staysure/data.py          CSV loading, Google Sheet loading, preprocessing, image loading
Model/src/staysure/model.py         CNN + metadata neural network
Model/src/staysure/metrics.py       MAE, RMSE, R2, MAPE, target transforms
Model/src/staysure/similarity.py    Similar-room embedding index
Model/src/staysure/explain.py       Prediction explanation helper
Model/src/staysure/config.py        Shared config and column names
```

Dataset files currently exist locally:

```text
Dataset/rooms_dataset.csv
Dataset/rooms/
```

At the last check:

- The CSV had 119 data rows.
- The image folder had 390 room image files.
- Referenced images in the CSV had 0 missing files.
- The target column is `rent_price`.

Expected CSV columns:

```text
id,img1,img2,img3,img4,img5,location,size_sqft,furnishing_type,cleanliness_score,amenities,rent_price,bhk
```

The code can also handle optional/default columns such as:

```text
bathroom_attached,description
```

## What Has Already Been Built

The model training pipeline is already implemented.

It supports:

- Loading data from a CSV file.
- Loading data from a public Google Sheet URL.
- Up to 5 images per listing.
- Missing image fallback using zero tensors.
- Metadata preprocessing with numeric scaling, one-hot categorical encoding, and amenities multi-label encoding.
- A CNN image branch.
- A dense metadata branch.
- Combined rent regression output.
- Log-transforming rent target during training.
- K-fold cross validation.
- Final full-dataset training.
- Saving trained artifacts.
- Building a similar-room embedding index.
- Running CLI inference after training.

The model architecture is in `Model/src/staysure/model.py`.

Current model structure:

- Input 1: `image_input`, shape `(5, 224, 224, 3)`.
- Image branch: TimeDistributed Conv2D blocks over up to 5 images.
- Visual embedding: pooled image features.
- Input 2: `metadata_input`, shape based on preprocessed metadata features.
- Metadata branch: dense layers.
- Combined branch: image + metadata features.
- Output: single rent regression value.

## Current Stage

We are at the final stage of Phase 1:

Train the first real model in Kaggle, inspect metrics, download artifacts, and then connect those artifacts to the backend.

The immediate next task is Kaggle training.

## Kaggle Training Plan

In Kaggle, there should be three inputs:

1. The StaySure repository or notebook working copy.
2. A dataset containing `rooms_dataset.csv`.
3. A dataset containing the `rooms/` image folder.

Recommended Kaggle paths:

```text
/kaggle/working/StaySure
/kaggle/input/staysure-dataset/rooms_dataset.csv
/kaggle/input/staysure-room-images/rooms
/kaggle/working/staysure_artifacts
```

If the Kaggle input names are different, only update the CSV and image paths in the commands.

## Kaggle Notebook Code

Use these cells in Kaggle.

### Cell 1: Define Paths

```python
from pathlib import Path

PROJECT_DIR = Path("/kaggle/working/StaySure")
CSV_PATH = Path("/kaggle/input/staysure-dataset/rooms_dataset.csv")
IMAGE_ROOT = Path("/kaggle/input/staysure-room-images/rooms")
OUTPUT_DIR = Path("/kaggle/working/staysure_artifacts")

print("Project exists:", PROJECT_DIR.exists(), PROJECT_DIR)
print("CSV exists:", CSV_PATH.exists(), CSV_PATH)
print("Image root exists:", IMAGE_ROOT.exists(), IMAGE_ROOT)
```

### Cell 2: Install Requirements

```python
!pip install -q -r /kaggle/working/StaySure/Model/requirements.txt
```

### Cell 3: Check Dataset

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head())

missing = []
for _, row in df.iterrows():
    for col in ["img1", "img2", "img3", "img4", "img5"]:
        value = str(row.get(col, "")).strip()
        if value and value.lower() not in {"nan", "none", "null"}:
            direct_path = IMAGE_ROOT / value
            name_path = IMAGE_ROOT / Path(value).name
            if not direct_path.exists() and not name_path.exists():
                missing.append((row.get("id"), col, value))

print("Missing images:", len(missing))
print("First missing:", missing[:10])
print(df["rent_price"].describe())
```

### Cell 4: Train Model

```python
!python /kaggle/working/StaySure/Model/train.py \
  --csv-path "/kaggle/input/staysure-dataset/rooms_dataset.csv" \
  --image-root "/kaggle/input/staysure-room-images/rooms" \
  --output-dir "/kaggle/working/staysure_artifacts" \
  --epochs 40 \
  --batch-size 8 \
  --folds 5 \
  --learning-rate 0.0005
```

For faster testing, use:

```python
!python /kaggle/working/StaySure/Model/train.py \
  --csv-path "/kaggle/input/staysure-dataset/rooms_dataset.csv" \
  --image-root "/kaggle/input/staysure-room-images/rooms" \
  --output-dir "/kaggle/working/staysure_artifacts" \
  --epochs 3 \
  --batch-size 8 \
  --folds 3 \
  --learning-rate 0.0005
```

### Cell 5: View Training Report

```python
import json
from pathlib import Path

report_path = Path("/kaggle/working/staysure_artifacts/training_report.json")

with open(report_path) as f:
    report = json.load(f)

report
```

### Cell 6: Test One Prediction

```python
!python /kaggle/working/StaySure/Model/predict.py \
  --model "/kaggle/working/staysure_artifacts/final_model.keras" \
  --preprocessor "/kaggle/working/staysure_artifacts/preprocessor.joblib" \
  --similar-index "/kaggle/working/staysure_artifacts/similar_rooms_index.csv" \
  --similar-embeddings "/kaggle/working/staysure_artifacts/similar_rooms_embeddings.npz" \
  --images "/kaggle/input/staysure-room-images/rooms/room1_img1.jpg" \
  --metadata '{"location":"Kripal Nagar, Bhilai, Chhattisgarh","bhk":"3","size_sqft":1200,"furnishing_type":"Semi-furnished","cleanliness_score":5.8,"amenities":"Kitchen, Parking, Water Supply"}'
```

### Cell 7: Zip Artifacts

```python
!cd /kaggle/working && zip -r staysure_artifacts.zip staysure_artifacts
```

Download:

```text
/kaggle/working/staysure_artifacts.zip
```

## Expected Training Outputs

After training, this folder should exist:

```text
/kaggle/working/staysure_artifacts
```

Expected files:

```text
best_model.keras
final_model.keras
preprocessor.joblib
training_report.json
similar_rooms_index.csv
similar_rooms_embeddings.npz
```

Meaning of each file:

- `best_model.keras`: best model from cross validation based on validation MAE.
- `final_model.keras`: final model trained on the full dataset.
- `preprocessor.joblib`: metadata preprocessor needed for inference.
- `training_report.json`: fold metrics and mean metrics.
- `similar_rooms_index.csv`: metadata rows used for similar room lookup.
- `similar_rooms_embeddings.npz`: vector embeddings for similar room search.

## How To Judge The Training Result

Open `training_report.json`.

Important metrics:

- `mae`: average rent prediction error in rupees.
- `rmse`: penalizes large errors more strongly.
- `r2`: closer to 1 is better.
- `mape`: percentage error.

Because the current dataset is small, model quality may be limited. If metrics are poor, do not immediately rewrite the model. First improve the dataset.

Likely dataset improvements:

- Add more rows.
- Make location values more consistent.
- Make `bhk` values consistent.
- Add better `bathroom_attached` values.
- Add better furnishing and amenities labels.
- Remove duplicate or low-quality listings.
- Increase rent diversity while keeping real market values.

## After Kaggle Training

Once artifacts are downloaded, the next step is to put them into:

```text
Model/artifacts/
```

Then test locally:

```bash
python Model/predict.py \
  --model Model/artifacts/final_model.keras \
  --preprocessor Model/artifacts/preprocessor.joblib \
  --similar-index Model/artifacts/similar_rooms_index.csv \
  --similar-embeddings Model/artifacts/similar_rooms_embeddings.npz \
  --images Dataset/rooms/room1_img1.jpg \
  --metadata '{"location":"Kripal Nagar, Bhilai, Chhattisgarh","bhk":"3","size_sqft":1200,"furnishing_type":"Semi-furnished","cleanliness_score":5.8,"amenities":"Kitchen, Parking, Water Supply"}'
```

If local inference works, move to backend integration.

## Backend Integration Goal

The backend should eventually expose:

```text
POST /predict
GET /similar-rooms
GET /history
POST /login
POST /signup
POST /train
```

For the first backend milestone, only `/predict` is necessary.

The backend prediction flow should be:

1. User uploads 1 to 5 images.
2. User submits metadata.
3. Backend saves temporary images.
4. Backend calls the trained TensorFlow model and preprocessor.
5. Backend returns:
   - predicted rent
   - confidence score
   - rent range
   - similar rooms
   - explanation

## Frontend Goal

The frontend should eventually allow users to:

- Upload room images.
- Enter room metadata.
- Submit for prediction.
- See predicted rent.
- See similar rooms.
- See explanation and confidence.
- Save prediction history after login.

Frontend is not the current priority until the model artifacts are trained and tested.

## Important Notes For Another LLM

Do not skip reading the existing code before changing anything.

Most of the model pipeline is already built. The next LLM should not recreate the model from scratch unless there is a clear reason.

The current priority order is:

1. Run Kaggle training.
2. Inspect `training_report.json`.
3. Download and place artifacts into `Model/artifacts/`.
4. Test `Model/predict.py`.
5. Build or connect the FastAPI `/predict` endpoint.
6. Build frontend only after backend prediction works.

If Kaggle paths fail, fix the paths first. Do not change model code unless the error shows an actual code bug.

If training runs but metrics are weak, improve dataset quality before making the neural network more complex.
