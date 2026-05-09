import streamlit as st
from transformers import pipeline
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Magic Story App", page_icon="🧸")

@st.cache_resource
def load_models():
    """Loads pre-trained models. Optimized for Streamlit Cloud stability."""
    # Image Captioning 
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Text Generation 
    gen_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Standard TTS 
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
    """Function 2: Generates a gentle and safe story for kids."""
    _, gen_model, _ = load_models()
    
    prompt = (
        f"<|user|>\n"
        f"Write a complete, short, and very gentle story for a 5-year-old child about {description}. "
        f"Structure: Start with 'Once upon a time', describe a happy scene, and end with a clear 'The end'. "
        f"Rules: Only use kind words. Children must share and be friends. No fighting or accidents. "
        f"Length: Exactly 3 to 4 simple sentences (around 60 words). <|assistant|>\n"
    )

    story_results = gen_model(
        prompt, 
        max_new_tokens=120,   
        min_new_tokens=60, 
        do_sample=True, 
        temperature=0.3,
        repetition_penalty=1.2
    )
  
    full_text = story_results[0]['generated_text']
    story_content = full_text.split("<|assistant|>\n")[-1].strip()

    if "." in story_content:
        story_content = story_content[:story_content.rindex(".")+1]

    return story_content


def text2audio(story_text):
    """Function 3: Converts text to speech."""
    _, _, tts_model = load_models()
    return tts_model(story_text)

# --- 3. MAIN UI ---
def main():
    """Function 4: The interactive UI."""
    st.title("🧸 Magic Storyteller")
    st.markdown("### Upload a picture to see and hear a story!")

    uploaded_file = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, width='stretch')

        if st.button("🌟 Start Magic"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Reading the picture...")
            desc = img2text(uploaded_file)
            progress_bar.progress(33)
            
            status_text.text("Creating a magic story...")
            story = text2story(desc)
            st.write(story)
            progress_bar.progress(66)
            
            status_text.text("Turning story into voice...")
            audio_data = text2audio(story)
            st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
            progress_bar.progress(100)
            
            status_text.text("Done!")
            st.balloons()

if __name__ == "__main__":
    main()
