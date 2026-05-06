import streamlit as st
from transformers import pipeline
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Magic Story App", page_icon="🧸")

@st.cache_resource
def load_models():
    """Loads pre-trained models. Optimized for Streamlit Cloud stability."""
    # Image Captioning [cite: 20, 21]
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Text Generation [cite: 23]
    gen_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Standard TTS [cite: 25]
    tts_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    return img_pipe, gen_pipe, tts_pipe

# --- 2. FUNCTIONS ---

def img2text(image_data):
    """Function 1: Extracts description from image."""
    img_model, _, _ = load_models()
    image = Image.open(image_data).convert("RGB")
    result = img_model(image)
    return result[0]["generated_text"]

def text2story(description):
    """Function 2: Generates a story (50-100 words) for kids."""
    _, gen_model, _ = load_models()
    
    prompt = (
        f"<|user|>\nTell a very simple story for a child about {description}. "
        f"Use easy words. Keep it between 60 to 80 words. <|assistant|>\n"
    )
    
    story_results = gen_model(
        prompt, 
        max_new_tokens=120, 
        min_new_tokens=60, 
        do_sample=True, 
        temperature=0.7
    )
    
    story = story_results[0]['generated_text'].split("<|assistant|>\n")[-1].strip()
    
    # Clean any potential AI prefixes
    if ":" in story and len(story.split(":")[0]) < 15:
        story = story.split(":")[-1].strip()
        
    return story[:500]

def text2audio(story_text):
    """Function 3: Converts text to speech."""
    _, _, tts_model = load_models()
    return tts_model(story_text)

# --- 3. MAIN UI ---

def main():
    """Function 4: The interactive UI."""
    st.title("🧸 Magic Storyteller")
    st.subtitle("Upload a picture to see and hear a story!")

    uploaded_file = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

        if st.button("🌟 Start Magic"):
            with st.spinner("Wait a moment..."):
                # 1. Image to Text
                desc = img2text(uploaded_file)
                
                # 2. Text to Story
                story = text2story(desc)
                
                st.write(story)
                
                # 3. Text to Audio
                audio_data = text2audio(story)
                st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
                
                st.balloons()

if __name__ == "__main__":
    main()
