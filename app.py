# Program title: Storytelling App for Kids
import streamlit as st
from transformers import pipeline
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Magic Story App", page_icon="🧸")

@st.cache_resource
def load_models():
    """
    Loads pre-trained models. 
    Optimized for stability on Streamlit Cloud. [cite: 72]
    """
    # Image Captioning: BLIP model [cite: 64]
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Text Generation: TinyLlama [cite: 66]
    gen_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Text-to-Speech: MMS TTS [cite: 68]
    tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    
    return img_pipe, gen_pipe, tts_pipe

# --- 2. MODULAR FUNCTIONS ---

def img2text(image_data):
    """Function 1: Processes image and generates a caption. [cite: 62]"""
    img_model, _, _ = load_models()
    image = Image.open(image_data).convert("RGB")
    result = img_model(image)
    return result[0]["generated_text"]

def text2story(description):
    """Function 2: Generates a gentle 50-100 word narrative. [cite: 57, 65]"""
    _, gen_model, _ = load_models()
    
    # Safe and gentle prompt for children 
    prompt = (
        f"<|user|>\n"
        f"Tell a very gentle, happy story for a toddler about {description}. "
        f"Rules: Use easy words like 'sun', 'happy', 'friend'. "
        f"The kids are kind and share toys. Make it 70 words long. <|assistant|>\n"
    )
    
    story_results = gen_model(
        prompt, 
        max_new_tokens=120,   
        min_new_tokens=60, 
        do_sample=True, 
        temperature=0.3, # Low temperature for polite and stable logic
        repetition_penalty=1.2
    )
    
    story = story_results[0]['generated_text'].split("<|assistant|>\n")[-1].strip()
    if "." in story:
        story = story[:story.rindex(".")+1]
    return story

def text2audio(story_text):
    """Function 3: Converts narrative to audio. [cite: 59, 67]"""
    _, _, tts_model = load_models()
    return tts_model(story_text)

# --- 3. MAIN UI ---

def main():
    """Function 4: Interactive UI for kids. [cite: 71, 74]"""
    # CSS to make the button bigger for kids
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            font-size: 24px; 
            height: 3em;
            width: 100%;
            border-radius: 20px;
            background-color: #FFD700;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧸 Magic Storyteller")
    # Larger font for the instruction
    st.markdown("### Upload a picture to hear a magic story!")

    uploaded_file = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

        if st.button("🌟 Start Magic"):
            with st.spinner("Writing your story..."):
                # Execution stages
                caption = img2text(uploaded_file)
                story = text2story(caption)
                
                # Display story text directly
                st.write(story)
                
                # Audio conversion and playback
                audio_data = text2audio(story)
                st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
                
                st.balloons()

if __name__ == "__main__":
    main()
