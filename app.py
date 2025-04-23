from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from transformers import (
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BlipForConditionalGeneration,
    BlipProcessor,
    GitForCausalLM,
    GitProcessor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer
)
import numpy as np

app = Flask(__name__)

# Load all captioning models and their components
models = {
    'vit-gpt2': {
        'model': VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning"),
        'feature_extractor': ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning"),
        'tokenizer': AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    },
    'blip': {
        'model': BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"),
        'processor': BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    },
    'git': {
        'model': GitForCausalLM.from_pretrained("microsoft/git-base-coco"),
        'processor': GitProcessor.from_pretrained("microsoft/git-base-coco")
    }
}

# Initialize translation model
translation_model_name = "Helsinki-NLP/opus-mt-en-ar"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move all models to device
for model_info in models.values():
    model_info['model'].to(device)
translation_model.to(device)

def translate_to_arabic(text):
    """Translate English text to Arabic using MarianMT model"""
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True)
    outputs = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def predict_with_vit_gpt2(image):
    pixel_values = models['vit-gpt2']['feature_extractor'](images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = models['vit-gpt2']['model'].generate(pixel_values, max_length=16, num_beams=4)
    preds = models['vit-gpt2']['tokenizer'].batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()

def predict_with_blip(image):
    inputs = models['blip']['processor'](images=image, return_tensors="pt").to(device)
    out = models['blip']['model'].generate(**inputs)
    return models['blip']['processor'].decode(out[0], skip_special_tokens=True)

def predict_with_git(image):
    inputs = models['git']['processor'](images=image, return_tensors="pt").to(device)
    out = models['git']['model'].generate(**inputs)
    return models['git']['processor'].decode(out[0], skip_special_tokens=True)

def predict_step(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    
    # Generate captions using all models
    captions = {
        'vit-gpt2': predict_with_vit_gpt2(image),
        'blip': predict_with_blip(image),
        'git': predict_with_git(image)
    }
    
    # Translate all captions to Arabic
    arabic_translations = {
        model_name: translate_to_arabic(caption)
        for model_name, caption in captions.items()
    }
    
    return {
        'english': captions,
        'arabic': arabic_translations
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        results = predict_step(file)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 