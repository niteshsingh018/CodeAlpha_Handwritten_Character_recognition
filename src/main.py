import os
from pathlib import Path

# Keep relative paths stable whether launched from the project root or src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Display dataset information
print("MNIST Dataset Loaded Successfully!")
print("-----------------------------------")

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

print("Testing images shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

print("Number of training images:", len(x_train))
print("Number of testing images:", len(x_test))

print("Image size:", x_train[0].shape)
print("First image label:", y_train[0])
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
print("\nPixel value range:")
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())
print("\nBefore preprocessing:")
print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)
print("Pixel range:", x_train.min(), "to", x_train.max())

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print("\nAfter preprocessing:")
print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

print("Pixel range:", x_train.min(), "to", x_train.max())

model = models.Sequential([
    
    # Convolutional Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # Pooling Layer 1
    layers.MaxPooling2D((2, 2)),

    # Convolutional Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Pooling Layer 2
    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into one-dimensional data
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(128, activation="relu"),

    # Output Layer
    layers.Dense(10, activation="softmax")
])

model.summary()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n================================")
print("CNN MODEL TRAINING COMPLETED")
print("================================")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()


# ------------------------------------------
# Step 2: Loss Graph
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()


# ------------------------------------------
# Step 3: Make Predictions
# ------------------------------------------

y_probability = model.predict(x_test)

y_pred = np.argmax(y_probability, axis=1)


# ------------------------------------------
# Step 4: Confusion Matrix
# ------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.title("MNIST Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()


# ------------------------------------------
# Step 5: Classification Report
# ------------------------------------------

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[str(i) for i in range(10)]
    )
)
os.makedirs("models", exist_ok=True)


# ------------------------------------------
# Step 1: Save the trained model
# ------------------------------------------

model_path = "models/mnist_cnn.keras"

model.save(model_path)

print("\n======================================")
print("MODEL SAVED SUCCESSFULLY")
print("======================================")
print("Model location:", model_path)


# ------------------------------------------
# Step 2: Load the saved model
# ------------------------------------------

loaded_model = tf.keras.models.load_model(model_path)

print("\n======================================")
print("MODEL LOADED SUCCESSFULLY")
print("======================================")


# ------------------------------------------
# Step 3: Test loaded model
# ------------------------------------------

loaded_loss, loaded_accuracy = loaded_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Loaded Model Accuracy:", loaded_accuracy)
print("Loaded Model Loss:", loaded_loss)