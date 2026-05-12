import re
from pathlib import Path
from typing import Iterable, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from tensorflow import keras

from .config import (
    CATEGORICAL_COLUMNS,
    IMAGE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


class MetadataPreprocessor:
    """Transforms room metadata into numeric vectors reusable during inference."""

    def __init__(self):
        self.column_transformer: Optional[ColumnTransformer] = None
        self.amenities_encoder = MultiLabelBinarizer()
        self.feature_dim: Optional[int] = None

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        df = normalize_dataframe(df, require_target=False)
        one_hot = _make_one_hot_encoder()
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_COLUMNS),
                ("cat", one_hot, CATEGORICAL_COLUMNS),
            ],
            remainder="drop",
        )
        base_features = self.column_transformer.fit_transform(df)
        amenities = self.amenities_encoder.fit_transform(
            df["amenities"].map(split_amenities)
        )
        self.feature_dim = int(base_features.shape[1] + amenities.shape[1])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.column_transformer is None:
            raise RuntimeError("MetadataPreprocessor must be fitted before transform.")
        df = normalize_dataframe(df, require_target=False)
        base_features = self.column_transformer.transform(df)
        amenities = self.amenities_encoder.transform(df["amenities"].map(split_amenities))
        features = np.hstack([np.asarray(base_features), amenities]).astype("float32")
        return features

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "MetadataPreprocessor":
        return joblib.load(path)


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def google_sheet_export_url(sheet_url: str) -> str:
    spreadsheet_match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not spreadsheet_match:
        raise ValueError("Could not find Google spreadsheet id in sheet URL.")
    spreadsheet_id = spreadsheet_match.group(1)
    gid_match = re.search(r"[?&#]gid=([0-9]+)", sheet_url)
    gid = gid_match.group(1) if gid_match else "0"
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        f"/export?format=csv&gid={gid}"
    )


def download_sheet_csv(sheet_url: str, output_path: Path) -> Path:
    export_url = google_sheet_export_url(sheet_url)
    response = requests.get(export_url, timeout=30)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path


def load_dataset(csv_path: Optional[Path] = None, sheet_url: Optional[str] = None) -> pd.DataFrame:
    if csv_path is None and sheet_url is None:
        raise ValueError("Provide either --csv-path or --sheet-url.")

    if csv_path is None:
        csv_path = Path("Dataset/rooms_from_sheet.csv")
        download_sheet_csv(sheet_url or "", csv_path)

    df = pd.read_csv(csv_path)
    return normalize_dataframe(df, require_target=True)


def normalize_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    for column in IMAGE_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    defaults = {
        "id": "",
        "location": "Unknown",
        "bhk": "Unknown",
        "furnishing_type": "Unknown",
        "amenities": "",
        "size_sqft": 0,
        "cleanliness_score": 5,
        "bathroom_attached": 0,
        "description": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    for column in CATEGORICAL_COLUMNS + ["amenities", "description"]:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().all():
            df[column] = df[column].fillna(0)
        else:
            df[column] = df[column].fillna(df[column].median())

    df["bathroom_attached"] = df["bathroom_attached"].map(_to_binary).astype("float32")

    if require_target:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Dataset must include target column '{TARGET_COLUMN}'.")
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] > 0)].reset_index(drop=True)

    return df


def split_amenities(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parts = re.split(r"[,|;/]+", str(value).lower())
    return sorted({part.strip() for part in parts if part.strip()})


def _to_binary(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "attached", "available"}:
        return 1
    return 0


def resolve_image_paths(row: pd.Series, image_root: Path) -> list[str]:
    paths = []
    for column in IMAGE_COLUMNS:
        value = str(row.get(column, "")).strip()
        paths.append(value)
    return paths


def load_room_image(value: str, image_root: Path, image_size: int) -> np.ndarray:
    if not value or value.lower() in {"nan", "none", "null"}:
        return np.zeros((image_size, image_size, 3), dtype="float32")

    if value.startswith("http://") or value.startswith("https://"):
        return _load_url_image(value, image_size)

    candidate_paths = []
    raw_path = Path(value)
    if raw_path.is_absolute():
        candidate_paths.append(raw_path)
    else:
        candidate_paths.append(image_root / raw_path)
        candidate_paths.append(image_root / raw_path.name)

    for path in candidate_paths:
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
                return image.astype("float32") / 255.0

    return np.zeros((image_size, image_size, 3), dtype="float32")


def _load_url_image(url: str, image_size: int) -> np.ndarray:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        buffer = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return np.zeros((image_size, image_size, 3), dtype="float32")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        return image.astype("float32") / 255.0
    except requests.RequestException:
        return np.zeros((image_size, image_size, 3), dtype="float32")


def build_image_tensor(
    rows: Iterable[pd.Series],
    image_root: Path,
    image_size: int,
    max_images: int,
) -> np.ndarray:
    room_images = []
    for row in rows:
        images = [
            load_room_image(value, image_root, image_size)
            for value in resolve_image_paths(row, image_root)[:max_images]
        ]
        while len(images) < max_images:
            images.append(np.zeros((image_size, image_size, 3), dtype="float32"))
        room_images.append(np.stack(images, axis=0))
    return np.stack(room_images, axis=0).astype("float32")


class RoomSequence(keras.utils.Sequence):
    """Small Keras-compatible sequence that loads images lazily per batch."""

    def __init__(
        self,
        df: pd.DataFrame,
        metadata_features: np.ndarray,
        image_root: Path,
        image_size: int,
        max_images: int,
        batch_size: int,
        targets: Optional[np.ndarray] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.metadata_features = metadata_features.astype("float32")
        self.image_root = image_root
        self.image_size = image_size
        self.max_images = max_images
        self.batch_size = batch_size
        self.targets = None if targets is None else targets.astype("float32")
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, batch_index: int):
        batch_ids = self.indices[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        rows = [self.df.iloc[int(idx)] for idx in batch_ids]
        images = build_image_tensor(rows, self.image_root, self.image_size, self.max_images)
        metadata = self.metadata_features[batch_ids]
        inputs = {"image_input": images, "metadata_input": metadata}
        if self.targets is None:
            return inputs
        return inputs, self.targets[batch_ids]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)
