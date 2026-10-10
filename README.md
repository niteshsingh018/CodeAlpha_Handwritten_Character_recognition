# Handwritten Digit Recognition (CNN)

A Flask web application that recognizes handwritten digits (0–9) using a Convolutional Neural Network trained on MNIST.

## Project structure

```text
handwritten-character-recognition/
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── data/
│   ├── raw/test.csv        # Optional local dataset (not required for training)
│   └── sample_images/      # Example images for CLI prediction
├── models/
│   └── mnist_cnn.keras     # Pre-trained model
├── reports/figures/        # Existing project figures
├── src/
│   ├── main.py             # Train, evaluate, and save the CNN
│   └── predict.py          # Predict from an example image
├── static/
│   ├── style.css
│   └── uploads/            # Runtime uploads
└── templates/
    └── index.html
```

## Setup (Windows PowerShell)

Python 3.10–3.12 is recommended for compatibility with TensorFlow.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Run the web application

From the project root (the folder containing `app.py`):

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Upload a clear image containing one handwritten digit. The included trained model is loaded automatically, so retraining is not required for normal use.

## Train the model again (optional)

```powershell
python src/main.py
```

The training script uses TensorFlow's MNIST dataset loader and saves the model to `models/mnist_cnn.keras`. On first use, TensorFlow may need an internet connection to download MNIST.

## CLI prediction (optional)

```powershell
python src/predict.py
```

## Notes

- `data/raw/test.csv` is approximately 49 MiB and is not required by the current training script. It is kept in the ZIP for your local use but excluded from Git tracking by `.gitignore`.
- The current model recognizes digits (MNIST, 0–9), not alphabetic characters. To recognize letters, train on a character dataset such as EMNIST and update the model's output classes.
- Uploaded images are stored in `static/uploads/`. Periodically remove old files if you deploy the app publicly.
