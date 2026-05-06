# Program title: Storytelling App
# Import part
import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Magic Story App", page_icon="🧸")

@st.cache_resource
def load_models():
    """
    Loads pre-trained models from Hugging Face. [cite: 5, 20]
    Using cached resources to optimize Streamlit performance. [cite: 28]
    """
    # Image Captioning [cite: 21]
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Text Generation (TinyLlama is stable and fast) [cite: 23]
    gen_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Text-to-Speech (High-quality female voice) [cite: 25]
    tts_pipe = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
    
    return img_pipe, gen_pipe, tts_pipe
    
# Function part
# --- Function 1: Image to Text
def img2text(image_data):
    img_model, _, _ = load_models()
    image = Image.open(image_data).convert("RGB")
    result = img_model(image)
    return result[0]["generated_text"]

# --- Function 2: Text to Story ---
def text2story(description):
    _, gen_model, _ = load_models()
    
# Specific instruction to use easy vocabulary for 3-10 year olds 
    prompt = (
        f"<|user|>\n"
        f"Write a very simple story for a toddler about {description}. "
        f"Rules: Use easy words like 'sun', 'happy', 'play'. "
        f"Use short sentences. No hard words. "
        f"Make it a happy story about 60 words long. <|assistant|>\n"
    )
    
    story_results = gen_model(
        prompt, 
        max_new_tokens=120,   
        min_new_tokens=60, 
        do_sample=True, 
        temperature=0.6,
        repetition_penalty=1.2
    )
    
    # Cleaning the output to show only the story
    full_text = story_results[0]['generated_text']
    story_content = full_text.split("<|assistant|>\n")[-1].strip()

    if "." in story_content:
        story_content = story_content[:story_content.rindex(".")+1]
        
    return story_content
    
# --- Function 3: Text to Audio ---
def text2audio(story_text):
    _, _, tts_model = load_models()
    return tts_model(story_text)
    
    return output

# --- Function 4: Main ---
def main():
    st.title("🧸 Magic Storyteller")
    st.write("Welcome! Upload a picture, and I will tell you a fun story!")

    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)

        # Trigger button
        if st.button("🌟 Start Magic"):
            with st.spinner("Making magic..."):
 
                # Execution stages 
                caption = img2text(uploaded_file)
                story = text2story(caption)
                
                st.subheader("Your Simple Story")
                st.write(story)
                
                # Audio part [cite: 16]
                audio_data = text2audio(story)
                st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
                
                st.balloons()

if __name__ == "__main__":
    main()
