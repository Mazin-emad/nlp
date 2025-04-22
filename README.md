# Image Captioning App

This is a simple web application that generates captions for images using a pre-trained deep learning model. The application uses the Vision Encoder Decoder model with ViT and GPT-2 for image captioning.

## Features

- Upload images through a web interface
- Generate captions automatically
- Modern and responsive UI
- Real-time preview of uploaded images

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the source code
   after setting up an py environment
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:

   ```
   http://localhost:5000
   ```

3. Click the "Select Image" button to choose an image file
4. The application will automatically generate a caption for the uploaded image

## Technical Details

The application uses:

- Flask for the web server
- PyTorch for deep learning
- Transformers library for the pre-trained model
- Vision Encoder Decoder model (ViT + GPT-2) for image captioning

## Note

The first time you run the application, it will download the pre-trained model weights, which might take a few minutes depending on your internet connection.

## License

This project is open source and available under the MIT License.
