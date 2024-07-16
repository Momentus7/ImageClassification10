import numpy as np
from tensorflow.keras.models import load_model
from src.utils import load_image

def predict_image(img_path):
    model = load_model("trained_model.h5")
    img_array = load_image(img_path)

    predictions = model.predict(img_array)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = class_names[np.argmax(predictions)]

    return {
        'predicted_class': predicted_class,
        'predictions': float(np.max(predictions))
    }
