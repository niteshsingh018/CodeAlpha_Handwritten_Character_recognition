from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ==========================================
# PHASE 7: HANDWRITTEN IMAGE PREDICTION
# ==========================================

# Load trained model
model = tf.keras.models.load_model(PROJECT_ROOT / "models" / "mnist_cnn.keras")

print("Model loaded successfully!")


# ------------------------------------------
# Image path
# ------------------------------------------

image_path = PROJECT_ROOT / "data" / "sample_images" / "Figure_1.png"


# ------------------------------------------
# Load image
# ------------------------------------------

image = Image.open(image_path)

# Convert to grayscale
image = image.convert("L")

# Resize to MNIST size
image = image.resize((28, 28))


# ------------------------------------------
# Convert image to NumPy array
# ------------------------------------------

image_array = np.array(image)


# ------------------------------------------
# Normalize pixel values
# ------------------------------------------

image_array = image_array.astype("float32") / 255.0


# ------------------------------------------
# Invert image if necessary
# ------------------------------------------

image_array = 1.0 - image_array


# ------------------------------------------
# Add dimensions for CNN
# ------------------------------------------

image_array = image_array.reshape(1, 28, 28, 1)


# ------------------------------------------
# Make prediction
# ------------------------------------------

prediction = model.predict(image_array, verbose=0)

predicted_digit = np.argmax(prediction[0])

confidence = np.max(prediction[0]) * 100


# ------------------------------------------
# Display result
# ------------------------------------------

print("\n======================================")
print("HANDWRITTEN DIGIT PREDICTION")
print("======================================")

print("Predicted Digit:", predicted_digit)
print("Confidence: {:.2f}%".format(confidence))


# ------------------------------------------
# Display image
# ------------------------------------------

plt.imshow(image_array[0].reshape(28, 28), cmap="gray")
plt.title(
    f"Prediction: {predicted_digit} "
    f"({confidence:.2f}%)"
)
plt.axis("off")
plt.show()