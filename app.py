import streamlit as st
from transformers import pipeline
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Magic Story App", page_icon="🧸")

@st.cache_resource
def load_models():
    """
    Loads pre-trained models. Using Qwen2.5-0.5B for extreme speed on CPU.
    """
    # Image Captioning
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Text Generation: Changed to a 0.5B parameter model for faster CPU performance
    gen_pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    
    # Standard TTS
    tts_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    return img_pipe, gen_pipe, tts_pipe


# --- 2. FUNCTIONS ---
def img2text(image_data):
    img_model, _, _ = load_models()
    image = Image.open(image_data).convert("RGB")
    result = img_model(image)
    return result[0]["generated_text"]

def text2story(description):
    _, gen_model, _ = load_models()
    
    # Optimized prompt for smaller models
    prompt = (
        f"Write a sweet, happy story for a 5-year-old child about {description}. "
        f"Make it 60 words. End with 'The end'."
    )
    
    # Using small max_new_tokens to ensure fast response on Streamlit CPU
    story_results = gen_model(
        prompt, 
        max_new_tokens=100,   
        min_new_tokens=55, # Keep meeting the 50-100 word requirement
        do_sample=True, 
        temperature=0.6,
        repetition_penalty=1.1
    )

    story_content = story_results[0]['generated_text']
    
    # Small models might include the prompt in output, let's clean it if necessary
    if prompt in story_content:
        story_content = story_content.replace(prompt, "").strip()

    if "." in story_content:
        story_content = story_content[:story_content.rindex(".")+1]

    return story_content


def text2audio(story_text):
    _, _, tts_model = load_models()
    return tts_model(story_text)

# --- 3. MAIN UI ---
def main():
    st.title("🧸 Magic Storyteller")
    st.markdown("### Fast & Magic Stories!")

    uploaded_file = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

        if st.button("🌟 Start Magic"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Caption
            status_text.text("Reading picture...")
            desc = img2text(uploaded_file)
            progress_bar.progress(33)

            # Step 2: Story (This should be much faster now)
            status_text.text("Creating story...")
            story = text2story(desc)
            st.info(story) 
            progress_bar.progress(66)
            
            # Step 3: Audio
            status_text.text("Generating voice...")
            audio_data = text2audio(story)
            st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
            progress_bar.progress(100)

            status_text.text("Done!")
            st.balloons()

if __name__ == "__main__":
    main()
