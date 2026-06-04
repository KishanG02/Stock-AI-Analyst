import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# Build Model
# -----------------------------
def build_model(look_back=60):

    model = Sequential([

        LSTM(
            128,
            return_sequences=True,
            input_shape=(look_back, 1)
        ),

        BatchNormalization(),
        Dropout(0.3),

        LSTM(
            64,
            return_sequences=True
        ),

        BatchNormalization(),
        Dropout(0.3),

        LSTM(
            32,
            return_sequences=False
        ),

        Dropout(0.2),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(),
        metrics=["mae"]
    )

    return model


# -----------------------------
# Train Model
# -----------------------------
def train_model(
    model,
    X_train,
    y_train,
    epochs=50,
    batch_size=32
):

    if len(X_train) == 0:
        raise ValueError("Training dataset is empty.")

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        ModelCheckpoint(
            "best_stock_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# -----------------------------
# Save Model
# -----------------------------
def save_model(model, path="stock_model.keras"):
    model.save(path)


# -----------------------------
# Load Model
# -----------------------------
def load_model(path="stock_model.keras"):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    return tf.keras.models.load_model(path)