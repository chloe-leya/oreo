
# Program title: Storytelling App

# Import part
import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Magic Story App", page_icon="🧸")

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
def img2text(image_data):
    img_model, _, _ = load_models()
    image = Image.open(image_data).convert("RGB")
    text = img_model(image)[0]["generated_text"]
    return text

# --- Function 2: Text to Story ---
def text2story(text):
    _, gen_model, _ = load_models()
    
    # We use a natural opening to guide the model without repeating commands[cite: 1]
    prompt = f"Once upon a time, there was {text}. It was a beautiful day and "
    
    # max_new_tokens ensures we add 50-80 new words to reach the total requirement[cite: 1]
    story_results = gen_model(
        prompt, 
        max_new_tokens=80, 
        do_sample=True, 
        temperature=0.8,
        pad_token_id=50256
    )
    
    full_text = story_results[0]['generated_text']
    final_story = full_text.strip()
    if "." in final_story:
        final_story = final_story[:final_story.rindex(".")+1]
        
    return final_story

# --- Function 3: Text to Audio ---
def text2audio(story_text):
    _, _, audio_model = load_models()
    return audio_model(story_text)


# --- Function 4: Main ---
def main():
    # Kid-friendly UI titles[cite: 1]
    st.title("🧸 Magic Storyteller")
    st.write("Welcome! Upload a picture, and I will tell you a fun story!")

    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Show the uploaded image
        st.image(uploaded_file, use_container_width=True)

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


