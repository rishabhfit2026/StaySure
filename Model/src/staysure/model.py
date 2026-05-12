import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_rent_model(
    metadata_dim: int,
    image_size: int = 224,
    max_images: int = 5,
    learning_rate: float = 1e-3,
) -> keras.Model:
    image_input = keras.Input(
        shape=(max_images, image_size, image_size, 3),
        name="image_input",
    )

    x = layers.TimeDistributed(
        layers.Conv2D(32, 3, padding="same", activation="relu")
    )(image_input)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    x = layers.TimeDistributed(
        layers.Conv2D(64, 3, padding="same", activation="relu")
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    x = layers.TimeDistributed(
        layers.Conv2D(128, 3, padding="same", activation="relu")
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    visual_embedding = layers.GlobalAveragePooling1D(name="visual_embedding")(x)
    visual_embedding = layers.Dense(128, activation="relu")(visual_embedding)
    visual_embedding = layers.Dropout(0.25)(visual_embedding)

    metadata_input = keras.Input(shape=(metadata_dim,), name="metadata_input")
    m = layers.Dense(128, activation="relu")(metadata_input)
    m = layers.BatchNormalization()(m)
    m = layers.Dropout(0.2)(m)
    m = layers.Dense(64, activation="relu")(m)

    combined = layers.Concatenate()([visual_embedding, m])
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.Dropout(0.25)(combined)
    joint_embedding = layers.Dense(64, activation="relu", name="joint_embedding")(combined)
    output = layers.Dense(1, activation="linear", name="rent_output")(joint_embedding)

    model = keras.Model(
        inputs={"image_input": image_input, "metadata_input": metadata_input},
        outputs=output,
        name="staysure_cnn_metadata_regressor",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def make_embedding_model(model: keras.Model) -> keras.Model:
    return keras.Model(inputs=model.inputs, outputs=model.get_layer("joint_embedding").output)


def set_global_determinism(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)

