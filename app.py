import streamlit as st
from transformers import pipeline
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Magic Story App", page_icon="🧸")

@st.cache_resource
def load_models():
    """
    Loads pre-trained models. TinyLlama-1.1B is used for a balance of speed and logic.
    """
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
    """Function 2: Generates the story and removes any unwanted titles/headers."""
    _, gen_model, _ = load_models()
    
    # YOUR SPECIFIC PROMPT (UNCHANGED)
    prompt = (
        f"<|user|>\n"
        f"Context: {description}. "
        f"Write a sweet and peaceful story for a 5-year-old child that strictly follows the Context provided. "
        f"The story must be 50-100 words and have a clear beginning and ending. "
        f"Guidelines: Use simple, happy words. The atmosphere is safe and gentle. "
        f"End with 'The end'. <|assistant|>\n"
    )
    
    # Optimized parameters for speed and word count requirements
    story_results = gen_model(
        prompt, 
        max_new_tokens=100,   
        min_new_tokens=55, 
        do_sample=True, 
        temperature=0.5,
        repetition_penalty=1.2
    )

    full_text = story_results[0]['generated_text']
    story_content = full_text.split("<|assistant|>\n")[-1].strip()

    # --- LOGIC TO REMOVE TITLE ---
    # This checks if the model hallucinated a "Title:" prefix and cuts it off
    if "Title" in story_content:
        # Split by "Title" and take the last part
        story_content = story_content.split("Title")[-1].lstrip(": ").strip()
    
    # Check for "Story:" prefix as well just in case
    if "Story:" in story_content:
        story_content = story_content.split("Story:")[-1].strip()

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
            st.info(story) # Show the story immediately
            progress_bar.progress(66)
            
            # Step 3: Audio Synthesis
            status_text.text("Turning story into voice...")
            audio_data = text2audio(story)
            st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
            progress_bar.progress(100)

            # Final celebration
            status_text.text("Done!")
            st.balloons()

if __name__ == "__main__":
    main()
