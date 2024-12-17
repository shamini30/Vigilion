import streamlit as st
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from gtts import gTTS
from googletrans import Translator
import os
import tempfile

# Load the fine-tuned CLIP model and processor from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "shamini30/VigilionApp"

# Load the model and processor directly from Hugging Face
try:
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
except Exception as e:
    st.error(f"Failed to load model from {model_name}. Error: {str(e)}")
    st.stop()

# Define the function to generate captions
def generate_caption(image):
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_text = outputs.logits_per_text

            # Decode the generated caption (select the most probable text tokens)
            caption = processor.tokenizer.decode(logits_per_text[0].argmax(dim=-1), skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Caption generation failed: {str(e)}")
        return "Unable to generate caption."

# Function to generate and play audio
def generate_audio(caption, language='en'):
    try:
        tts = gTTS(text=caption, lang=language, slow=False)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_audio_file:
            tts.save(tmp_audio_file.name)
            return tmp_audio_file.name
    except Exception as e:
        st.error(f"Audio generation failed: {str(e)}")
        return None

# Streamlit interface
st.title("Vigilion: Your Personal Smart Vision Assistant")
st.write("Let us be your eye and guide you!")

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
language = st.selectbox("Select your language:", list(languages.keys()))

# Upload image
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Generate a caption using the fine-tuned CLIP model
        caption = generate_caption(image)
        st.subheader("Caption Generated:")
        st.write(caption)

        # Translate the caption
        translator = Translator()
        translated_caption = translator.translate(caption, dest=languages[language]).text

        # Generate audio in the selected language
        audio_file = generate_audio(translated_caption, languages[language])

        if audio_file:
            st.audio(audio_file, format="audio/mp3")

        # Clean up temporary files after use
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

    except Exception as e:
        st.error(f"Error processing the image or generating output: {str(e)}")
