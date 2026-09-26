# Image Recognition System

A Flask web application that uses a pretrained TensorFlow/Keras MobileNetV2
model to classify uploaded images with ImageNet labels and confidence scores.

## Features

- Upload JPG, JPEG, or PNG images
- Drag-and-drop upload area with client-side image preview
- Server-side file validation and a 5 MB upload limit
- MobileNetV2 classification with ImageNet pretrained weights
- Predicted object name and confidence percentage
- Loading state while an image is being analyzed
- In-memory Recent Predictions history for the five most recent results
- Responsive interface for desktop and mobile screens
- Temporary upload cleanup after each request

## Technologies

- Python 3.13+
- Flask
- TensorFlow/Keras
- MobileNetV2
- Pillow
- NumPy
- Jinja2 templates

## Installation

### Replit

The project is configured to run with the Replit workflow:

```bash
python app.py
```

The Flask server listens on `0.0.0.0:5000`, which makes it available through
the Replit preview.

### Local setup

1. Install Python 3.13 or a compatible Python version.
2. Install the project dependencies:

   ```bash
   pip install -e .
   ```

3. Start the application:

   ```bash
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in a browser.

The first startup may download the MobileNetV2 ImageNet weights.

## Usage

1. Open the application in a browser.
2. Choose a JPG, JPEG, or PNG image, or drag one into the upload area.
3. Review the client-side preview.
4. Select **Predict Image**.
5. Review the predicted class and confidence percentage.
6. Use **Try Another Image** to classify a different image.

Prediction history is stored only in the running process. It is limited to the
five most recent successful predictions and is cleared when the application
restarts. No database or login system is used.

## Project Structure

```text
.
├── app.py                 # Flask application and prediction route
├── templates/
│   └── index.html         # Upload interface, preview, result, and history UI
├── pyproject.toml         # Python project metadata and dependencies
├── uv.lock                # Locked dependency versions
├── .replit               # Replit runtime and workflow configuration
├── .gitignore             # Ignored local, generated, and sensitive files
└── README.md              # Project documentation
```

## Security and Privacy Notes

- Uploaded files are checked for supported extensions and valid image content.
- Filenames are sanitized and temporary upload files are removed after
  processing.
- Uploads are not committed to the repository.
- No API keys, passwords, or other credentials are required by the project.