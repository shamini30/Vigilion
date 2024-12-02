import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from gtts import gTTS
from googletrans import Translator
import os
import tempfile

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_generation_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Define the function to generate captions
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = caption_generation_model.generate(**inputs, num_beams=5)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Function to generate and play audio
def generate_audio(caption, language='en'):
    tts = gTTS(text=caption, lang=language, slow=False)
    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(delete=False) as tmp_audio_file:
        tts.save(tmp_audio_file.name)
        return tmp_audio_file.name

# Streamlit interface
st.title("Vigilion: Your Personal Smart Vision Assistant")
st.write("Upload an image to hear the description:")

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

# User selects language
language = st.selectbox("Select Language for Audio", list(languages.keys()))

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Generate caption
    caption = generate_caption(image)
    st.subheader("There is:")
    st.write(caption)
    
    # Translate the caption
    translator = Translator()
    translated_caption = translator.translate(caption, dest=languages[language]).text
    
    # Generate audio in the selected language
    audio_file = generate_audio(translated_caption, languages[language])
    
    # Play the audio file
    st.audio(audio_file, format="audio/mp3")
    
    # Optionally delete the temporary audio file after use
    os.remove(audio_file)
