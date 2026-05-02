
# Program title: Storytelling App

# Import part
import streamlit as st
from transformers import pipeline

# Helper function to load models safely and prevent memory crashes on Streamlit Cloud
@st.cache_resource
def load_models():
    # Using the required model for image captioning
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    gen_pipe = pipeline("text-generation", model="gpt2")
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    return img_pipe, gen_pipe, audio_pipe

# Function part
# --- Function 1: Image to Text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text
def img2text(image_data):
    img_model, _, _ = load_models()
    text = img_model(image_data)[0]["generated_text"]
    return text

# --- Function 2: Text to Story ---
def text2story(text):
    _, gen_model, _ = load_models()
    
    # Prompt tailored for kids aged 3-10 using easy words
    prompt = f"Write a simple, happy story for a 5-year-old child about: {text}. Use easy words. Once upon a time,"
    
    # Keeping the story length between 50-100 words
    story_results = gen_model(prompt, max_length=100, min_length=50, do_sample=True)
    story_text = story_results[0]['generated_text']
    return story_text

# --- Function 3: Text to Audio ---
def text2audio(story_text):
    _, _, audio_model = load_models()
    audio_data = audio_model(story_text)
    return audio_data

# --- Function 4: Main ---
def main():
    # Kid-friendly UI titles[cite: 1]
    st.title("🧸 Magic Storyteller")
    st.write("Welcome! Upload a picture, and I will tell you a fun story!")

    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Show the uploaded image
        st.image(uploaded_file, caption="Your Picture", use_container_width=True)

        # Trigger button
        if st.button("Generate Magic Story"):
            with st.spinner("Making magic..."):
                
                # Stage 1: Get Text
                caption = img2text(uploaded_file)
                st.info(f"I see: {caption}")
                
                # Stage 2: Get Story
                story = text2story(caption)
                st.success("Here is your story:")
                st.write(story)
                
                # Stage 3: Get Audio
                audio_data = text2audio(story)
                st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
                
                # Fun balloon animation for kids[cite: 1]
                st.balloons()


# Run the application

if __name__ == "__main__":
    main()


