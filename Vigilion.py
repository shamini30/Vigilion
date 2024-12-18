import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from gtts import gTTS
from googletrans import Translator
import os
import tempfile

# Initialize device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load BLIP Model ---
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()
except Exception as e:
    st.error(f"Failed to load the BLIP model: {str(e)}")
    st.stop()

# --- Caption Generation Function ---
def generate_caption(image):
    try:
        # Preprocess the image for BLIP
        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, num_beams=5)  # Beam search for better results
        caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error during caption generation: {e}")
        return None

# --- Generate Audio Function ---
def generate_audio(caption, language='en'):
    try:
        tts = gTTS(text=caption, lang=language, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
            tts.save(tmp_audio_file.name)
            return tmp_audio_file.name
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# --- Streamlit Interface ---
st.title("Vigilion: Your Personal Smart Vision Assistant")
st.write("Let us guide you with smart visual capabilities.")

# Language Selection
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
language = st.selectbox("Choose Language", list(languages.keys()))

# Image Upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Load the image using PIL
    img = Image.open(uploaded_image)

    # Ensure the image is in RGB format
    if img.mode != "RGB":
        img = img.convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    caption = generate_caption(img)
    if caption:
        st.write("Generated Caption:", caption)

        # Translate the caption
        translator = Translator()
        translated_caption = translator.translate(caption, dest=languages[language]).text
        st.write("Translated Caption:", translated_caption)

        # Generate and play audio
        audio_file = generate_audio(translated_caption, language=languages[language])
        if audio_file:
            st.audio(audio_file)
    else:
        st.error("Caption generation failed.")
