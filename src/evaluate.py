import numpy as np
from tensorflow.keras.models import load_model
from src.data_ingestion import load_cifar10_data
from src.data_preprocessing import preprocess_data

def evaluate_model():
    model = load_model("trained_model.h5")
    
    (X_train, y_train), (X_test, y_test) = load_cifar10_data()
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
