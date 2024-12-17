import streamlit as st
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from gtts import gTTS
from googletrans import Translator
import os
import tempfile

# Load the fine-tuned CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "shamini30/VigilionApp"


# Check if the model path exists
if not os.path.exists(model_path):
    st.error(f"Model path '{model_path}' does not exist. Please check the path.")

# Load the model and processor from the specified local path
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Define the function to generate captions
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = caption_generation_model(**inputs)
        logits_per_text = outputs.logits_per_text

        # Decode the generated caption (select the most probable text tokens)
        caption = processor.tokenizer.decode(logits_per_text[0].argmax(dim=-1), skip_special_tokens=True)
    return caption

# Function to generate and play audio
def generate_audio(caption, language='en'):
    tts = gTTS(text=caption, lang=language, slow=False)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_audio_file:
        tts.save(tmp_audio_file.name)
        return tmp_audio_file.name

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

    # Play the audio file
    st.audio(audio_file, format="audio/mp3")

    # Clean up temporary files after use
    os.remove(audio_file)
