import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model" / "src"))

from staysure.config import TARGET_COLUMN, TrainingConfig
from staysure.data import MetadataPreprocessor, RoomSequence, load_dataset
from staysure.metrics import inverse_target, regression_metrics, transform_target
from staysure.model import build_rent_model, set_global_determinism
from staysure.similarity import build_similarity_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StaySure AI rent prediction model.")
    parser.add_argument("--csv-path", type=Path, default=None, help="Local CSV path.")
    parser.add_argument("--sheet-url", type=str, default=None, help="Public Google Sheet URL.")
    parser.add_argument("--image-root", type=Path, required=True, help="Directory containing room images.")
    parser.add_argument("--output-dir", type=Path, default=Path("Model/artifacts"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-final-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        folds=args.folds,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    set_global_determinism(config.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path=args.csv_path, sheet_url=args.sheet_url)
    if len(df) < 2:
        raise ValueError("Need at least 2 valid rows with positive rent_price to train.")

    fold_metrics = run_cross_validation(df, args.image_root, args.output_dir, config)
    report = {
        "rows": int(len(df)),
        "fold_metrics": fold_metrics,
        "mean_metrics": _mean_metrics(fold_metrics),
    }

    if not args.no_final_train:
        final_artifacts = train_final_model(df, args.image_root, args.output_dir, config)
        report["final_artifacts"] = final_artifacts

    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_cross_validation(
    df: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    n_splits = min(config.folds, len(df))
    if n_splits < 2:
        return []

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    fold_metrics = []
    best_mae = float("inf")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df), start=1):
        print(f"\nStarting fold {fold}/{n_splits}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        preprocessor = MetadataPreprocessor().fit(train_df)
        train_meta = preprocessor.transform(train_df)
        val_meta = preprocessor.transform(val_df)

        y_train = transform_target(
            train_df[TARGET_COLUMN].to_numpy(dtype="float32"),
            config.target_log_transform,
        )
        y_val_raw = val_df[TARGET_COLUMN].to_numpy(dtype="float32")

        model = build_rent_model(
            metadata_dim=preprocessor.feature_dim or train_meta.shape[1],
            image_size=config.image_size,
            max_images=config.max_images,
            learning_rate=config.learning_rate,
        )

        train_seq = RoomSequence(
            train_df,
            train_meta,
            image_root,
            config.image_size,
            config.max_images,
            config.batch_size,
            targets=y_train,
            shuffle=True,
            seed=config.seed + fold,
        )
        val_seq = RoomSequence(
            val_df,
            val_meta,
            image_root,
            config.image_size,
            config.max_images,
            config.batch_size,
            targets=transform_target(y_val_raw, config.target_log_transform),
            shuffle=False,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ]
        model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        pred = model.predict(val_seq, verbose=0).reshape(-1)
        pred_rent = inverse_target(pred, config.target_log_transform)
        metrics = regression_metrics(y_val_raw, pred_rent)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        print(f"Fold {fold} metrics: {metrics}")

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            model.save(output_dir / "best_model.keras")
            preprocessor.save(output_dir / "preprocessor.joblib")

    return fold_metrics


def train_final_model(
    df: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, str]:
    print("\nTraining final model on full dataset")
    preprocessor = MetadataPreprocessor().fit(df)
    metadata = preprocessor.transform(df)
    targets = transform_target(
        df[TARGET_COLUMN].to_numpy(dtype="float32"),
        config.target_log_transform,
    )
    model = build_rent_model(
        metadata_dim=preprocessor.feature_dim or metadata.shape[1],
        image_size=config.image_size,
        max_images=config.max_images,
        learning_rate=config.learning_rate,
    )
    sequence = RoomSequence(
        df,
        metadata,
        image_root,
        config.image_size,
        config.max_images,
        config.batch_size,
        targets=targets,
        shuffle=True,
        seed=config.seed,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    model.fit(sequence, epochs=config.epochs, callbacks=callbacks, verbose=1)

    final_model_path = output_dir / "final_model.keras"
    preprocessor_path = output_dir / "preprocessor.joblib"
    model.save(final_model_path)
    preprocessor.save(preprocessor_path)

    index_path, embeddings_path = build_similarity_index(
        model=model,
        df=df,
        metadata_features=metadata,
        image_root=image_root,
        image_size=config.image_size,
        max_images=config.max_images,
        batch_size=config.batch_size,
        output_dir=output_dir,
    )

    return {
        "model": str(final_model_path),
        "preprocessor": str(preprocessor_path),
        "similar_rooms_index": str(index_path),
        "similar_rooms_embeddings": str(embeddings_path),
    }


def _mean_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not fold_metrics:
        return {}
    metric_names = [name for name in fold_metrics[0].keys() if name != "fold"]
    return {
        name: float(np.mean([fold[name] for fold in fold_metrics]))
        for name in metric_names
    }


if __name__ == "__main__":
    main()

