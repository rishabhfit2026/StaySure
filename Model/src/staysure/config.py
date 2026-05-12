from dataclasses import dataclass


IMAGE_COLUMNS = ["img1", "img2", "img3", "img4", "img5"]
NUMERIC_COLUMNS = ["size_sqft", "cleanliness_score", "bathroom_attached"]
CATEGORICAL_COLUMNS = ["location", "bhk", "furnishing_type"]
TEXT_COLUMNS = ["amenities"]
TARGET_COLUMN = "rent_price"


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = 224
    max_images: int = 5
    batch_size: int = 16
    epochs: int = 25
    folds: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    target_log_transform: bool = True

