import streamlit as st
from transformers import pipeline
from PIL import Image

# --- 1. CONFIGURATION ---
# Set the page title and a kid-friendly icon for the browser tab
st.set_page_config(page_title="Magic Story App", page_icon="🧸")

@st.cache_resource
def load_models():
    """
    Loads pre-trained models using the Hugging Face pipeline.
    @st.cache_resource ensures models are loaded only once to save memory and time.
    """
    # Image Captioning：BLIP model for generating a text description from an uploaded image
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Text Generation: Changed to TinyStories-33M for faster CPU inference and kid-friendly content
    gen_pipe = pipeline("text-generation", model="roneneldan/TinyStories-33M")
    
    # Standard TTS： Facebook's MMS model for converting the generated story into natural speech
    tts_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    return img_pipe, gen_pipe, tts_pipe


# --- 2. FUNCTIONS ---
def img2text(image_data):
    """Function 1: Extracts description from image."""
    img_model, _, _ = load_models()
    # Convert image to RGB to ensure compatibility with the BLIP model
    image = Image.open(image_data).convert("RGB")
    result = img_model(image)
    return result[0]["generated_text"]

def text2story(description):
    """Function 2: Generates a gentle and safe story for kids using TinyStories."""
    _, gen_model, _ = load_models()
    
    # TinyStories model works best with a simple narrative prompt
    prompt = f"Once upon a time, there was {description}. The children were very happy and "
    
    # Generate story with parameters optimized for the 33M model
    story_results = gen_model(
        prompt, 
        max_new_tokens=85,    # Keeps it within the 50-100 word requirement
        do_sample=True, 
        temperature=0.7, 
        top_p=0.95,
        repetition_penalty=1.1
    )

    story_content = story_results[0]['generated_text']

    # Ensure the story ends at a full sentence
    if "." in story_content:
        story_content = story_content[:story_content.rindex(".")+1]
        
    # Manually append the ending to ensure completeness as requested
    if "The end" not in story_content:
        story_content += " They all had a wonderful day. The end."

    return story_content


def text2audio(story_text):
    """Function 3: Converts text to speech."""
    _, _, tts_model = load_models()
    # Ensure standard task name is used
    return tts_model(story_text)

# --- 3. MAIN UI ---
def main():
    """Function 4: The interactive UI."""
    st.title("🧸 Magic Storyteller")
    st.markdown("### Upload a picture to see and hear a story!")

    uploaded_file = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

        if st.button("🌟 Start Magic"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Image Captioning
            status_text.text("Reading the picture...")
            desc = img2text(uploaded_file)
            progress_bar.progress(33)

            # Step 2: Story Generation
            status_text.text("Creating a magic story...")
            story = text2story(desc)
            st.info(story) # Show story text first to improve UX
            progress_bar.progress(66)
            
            # Step 3: Audio Synthesis
            status_text.text("Turning story into voice...")
            audio_data = text2audio(story)
            st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
            progress_bar.progress(100)

            # Final celebration effect for kids
            status_text.text("Done!")
            st.balloons()

if __name__ == "__main__":
    main()
