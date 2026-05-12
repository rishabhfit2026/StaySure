from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .data import RoomSequence
from .model import make_embedding_model


def build_similarity_index(
    model,
    df: pd.DataFrame,
    metadata_features: np.ndarray,
    image_root: Path,
    image_size: int,
    max_images: int,
    batch_size: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    sequence = RoomSequence(
        df=df,
        metadata_features=metadata_features,
        image_root=image_root,
        image_size=image_size,
        max_images=max_images,
        batch_size=batch_size,
        targets=None,
        shuffle=False,
    )
    embedding_model = make_embedding_model(model)
    embeddings = embedding_model.predict(sequence, verbose=0)

    index_df = df[
        [
            "id",
            "location",
            "bhk",
            "size_sqft",
            "furnishing_type",
            "cleanliness_score",
            "amenities",
            "rent_price",
            "img1",
        ]
    ].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "similar_rooms_index.csv"
    embeddings_path = output_dir / "similar_rooms_embeddings.npz"
    index_df.to_csv(index_path, index=False)
    np.savez_compressed(embeddings_path, embeddings=embeddings.astype("float32"))
    return index_path, embeddings_path


def find_similar_rooms(
    query_embedding: np.ndarray,
    index_df: pd.DataFrame,
    index_embeddings: np.ndarray,
    query_metadata: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query_embedding = np.asarray(query_embedding).reshape(1, -1)
    visual_scores = cosine_similarity(query_embedding, index_embeddings)[0]

    metadata_scores = np.array(
        [_metadata_similarity(query_metadata, row) for _, row in index_df.iterrows()],
        dtype="float32",
    )
    combined_scores = 0.7 * visual_scores + 0.3 * metadata_scores
    best_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for idx in best_indices:
        row = index_df.iloc[int(idx)]
        results.append(
            {
                "id": str(row.get("id", "")),
                "location": str(row.get("location", "")),
                "bhk": str(row.get("bhk", "")),
                "rent_price": float(row.get("rent_price", 0)),
                "image": str(row.get("img1", "")),
                "similarity_percentage": round(float(combined_scores[idx]) * 100, 2),
                "visual_similarity": round(float(visual_scores[idx]) * 100, 2),
            }
        )
    return results


def _metadata_similarity(query_metadata: dict[str, Any], row: pd.Series) -> float:
    score = 0.0
    weight = 0.0

    for key, key_weight in [("location", 0.45), ("bhk", 0.3), ("furnishing_type", 0.15)]:
        weight += key_weight
        if str(query_metadata.get(key, "")).lower() == str(row.get(key, "")).lower():
            score += key_weight

    weight += 0.1
    try:
        query_size = float(query_metadata.get("size_sqft", 0) or 0)
        row_size = float(row.get("size_sqft", 0) or 0)
        if max(query_size, row_size) > 0:
            score += 0.1 * (1 - min(abs(query_size - row_size) / max(query_size, row_size), 1))
    except (TypeError, ValueError):
        pass

    return score / max(weight, 1e-6)

