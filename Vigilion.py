import streamlit as st
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from gtts import gTTS
from googletrans import Translator
import os
import tempfile

# Define device for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model path
model_path = "shamini30/VigilionApp/fine_tuned_clip_epoch_3"

# Clear CUDA cache
torch.cuda.empty_cache()

# Load the fine-tuned CLIP model and processor
try:
    model = CLIPModel.from_pretrained(model_path, safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")
    st.stop()

# Function to generate captions
def generate_caption(image):
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_text = outputs.logits_per_text
            caption = processor.tokenizer.decode(logits_per_text[0].argmax(dim=-1), skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Failed to generate captions: {str(e)}")
        return "Caption generation failed."

# Function to generate and play audio
def generate_audio(caption, language='en'):
    tts = gTTS(text=caption, lang=language, slow=False)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_audio_file:
        tts.save(tmp_audio_file.name)
        return tmp_audio_file.name

# Streamlit Interface
st.title("Vigilion: Your Personal Smart Vision Assistant")
st.write("Let us guide you with smart visual capabilities.")

# Language selection
languages = {
    'English': 'en',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Urdu': 'ur',
    'Gujarati': 'gu',
    'Malayalam': 'ml',
    'Kannada': 'kn'
}

# User selects a language
language = st.selectbox("Choose Language", list(languages.keys()))

# Upload an image
uploaded_image = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img)

    # Generate a caption using the fine-tuned CLIP model
    caption = generate_caption(img)
    st.write("Generated Caption:", caption)

    # Translate the caption
    translator = Translator()
    translated_caption = translator.translate(caption, dest=languages[language]).text

    # Generate and play audio in the selected language
    audio_file = generate_audio(translated_caption)

    if os.path.exists(audio_file):
        st.audio(audio_file)
