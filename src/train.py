import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data_ingestion import load_cifar10_data
from src.data_preprocessing import preprocess_data, split_data
from src.model import build_model

def train_model():
    (X_train, y_train), (X_test, y_test) = load_cifar10_data()
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)
    X_train, X_valid, y_train, y_valid = split_data(X_train, y_train)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.12,
        height_shift_range=0.12,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        shear_range=10,
        channel_shift_range=0.1,
    )

    model = build_model(X_train.shape[1:])
    model.summary()

    batch_size = 64
    epochs = 100

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=[reduce_lr, early_stopping],
        verbose=2
    )

    model.save("trained_model.h5")

    return {
        'message': 'Training completed',
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'train_accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1]
    }
