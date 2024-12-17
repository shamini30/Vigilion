import streamlit as st
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from gtts import gTTS
from googletrans import Translator
import os
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "shamini30/VigilionApp"

# Load the fine-tuned CLIP model and processor
try:
    # Load the model and processor directly with HuggingFace safetensors support
    model = CLIPModel.from_pretrained(model_name, safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")
    st.stop()

# Define the function to generate captions
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

def generate_audio(caption, language='en'):
    tts = gTTS(text=caption, lang=language, slow=False)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_audio_file:
        tts.save(tmp_audio_file.name)
        return tmp_audio_file.name

# Streamlit UI
st.title("Vigilion: Your Personal Smart Vision Assistant")
st.write("Let us guide you with smart visual capabilities.")

language = st.selectbox("Choose Language", ['English', 'Hindi', 'Bengali'])
uploaded_image = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img)

    caption = generate_caption(img)
    st.write("Generated Caption:", caption)

    translator = Translator()
    translated_caption = translator.translate(caption, dest='en').text

    audio_file = generate_audio(translated_caption)

    if os.path.exists(audio_file):
        st.audio(audio_file)

