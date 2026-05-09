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
    # Text Generation： TinyLlama model for creative text generation (optimized for small-scale deployment)
    gen_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
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
    """Function 2: Generates a gentle and safe story for kids."""
    _, gen_model, _ = load_models()
    
    prompt = (
        f"<|user|>\n"
        f"Create a very simple, 3-sentence happy story for a 5-year-old about {description}. "
        f"Sentence 1: Start with 'Once upon a time' and describe the friends playing. "
        f"Sentence 2: Describe a kind action, like sharing a toy or a smile. "
        f"Sentence 3: End the story with: 'They all had a wonderful day. The end.' "
        f"Rules: No bad news, no accidents, and no loud noises. Be sweet and complete. <|assistant|>\n"
    )
    
    # Sampling parameters tuned for creative yet stable output (low temperature = more polite)
    story_results = gen_model(
        prompt, 
        max_new_tokens=120,   
        min_new_tokens=60, 
        do_sample=True, 
        temperature=0.3,
        repetition_penalty=1.2
    )

    # Clean the output to ensure it only contains the assistant's generated story
    full_text = story_results[0]['generated_text']
    story_content = full_text.split("<|assistant|>\n")[-1].strip()

    # Ensure the story ends at a full sentence
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

            # Step 1: Image Captioning
            status_text.text("Reading the picture...")
            desc = img2text(uploaded_file)
            progress_bar.progress(33)

            # Step 2: Story Generation
            status_text.text("Creating a magic story...")
            story = text2story(desc)
            st.write(story)
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
