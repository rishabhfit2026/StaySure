import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model" / "src"))

from staysure.data import MetadataPreprocessor, RoomSequence, normalize_dataframe
from staysure.explain import explain_prediction
from staysure.metrics import confidence_from_neighbors, inverse_target
from staysure.model import make_embedding_model
from staysure.similarity import find_similar_rooms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StaySure rent prediction inference.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--similar-index", type=Path, default=None)
    parser.add_argument("--similar-embeddings", type=Path, default=None)
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--metadata", type=str, required=True, help="JSON metadata object.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata)
    row = metadata.copy()
    for idx in range(args.max_images):
        row[f"img{idx + 1}"] = args.images[idx] if idx < len(args.images) else ""

    df = normalize_dataframe(pd.DataFrame([row]), require_target=False)
    preprocessor = MetadataPreprocessor.load(args.preprocessor)
    metadata_features = preprocessor.transform(df)
    model = keras.models.load_model(args.model)

    sequence = RoomSequence(
        df=df,
        metadata_features=metadata_features,
        image_root=Path("."),
        image_size=args.image_size,
        max_images=args.max_images,
        batch_size=1,
        targets=None,
        shuffle=False,
    )
    predicted_log_rent = model.predict(sequence, verbose=0).reshape(-1)[0]
    predicted_rent = float(inverse_target(np.array([predicted_log_rent]), True)[0])

    similar_rooms = []
    similarities = np.array([])
    if args.similar_index and args.similar_embeddings:
        index_df = pd.read_csv(args.similar_index)
        index_embeddings = np.load(args.similar_embeddings)["embeddings"]
        embedding_model = make_embedding_model(model)
        query_embedding = embedding_model.predict(sequence, verbose=0)
        similar_rooms = find_similar_rooms(
            query_embedding=query_embedding,
            index_df=index_df,
            index_embeddings=index_embeddings,
            query_metadata=metadata,
            top_k=5,
        )
        similarities = np.array(
            [room["similarity_percentage"] / 100 for room in similar_rooms],
            dtype="float32",
        )

    similar_rents = [room["rent_price"] for room in similar_rooms]
    rent_std = float(np.std(similar_rents)) if similar_rents else predicted_rent * 0.15
    confidence = confidence_from_neighbors(similarities, predicted_rent, rent_std)
    margin = max(predicted_rent * (1 - confidence) * 0.5, predicted_rent * 0.08)

    result = {
        "predicted_rent": round(predicted_rent, 2),
        "confidence_score": round(confidence, 3),
        "estimated_range": {
            "min": round(max(predicted_rent - margin, 0), 2),
            "max": round(predicted_rent + margin, 2),
        },
        "similar_rooms": similar_rooms,
        "explanation": explain_prediction(metadata, predicted_rent, similar_rooms),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

