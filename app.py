# Program title: Storytelling App
# Import part
import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Magic Story App", page_icon="🧸")

@st.cache_resource
def load_models():
    """
    Loads and caches transformers pipelines for efficiency.
    Ensures models are only loaded once to save memory on Streamlit Cloud.
    """
    # Image Captioning [cite: 21]
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Text Generation [cite: 23]
    gen_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Text-to-Speech [cite: 25]
    tts_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
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
        f"You are a world-class storyteller for 5-year-old children. "
        f"Write a magical, happy story about: {description}. "
        f"Include cheerful sounds, bright colors, and simple feelings. "
        f"Use very easy words. Make it around 70-80 words. "
        f"Start the story directly. <|assistant|>\n"
    )
    
    story_results = gen_model(
        prompt, 
        max_new_tokens=120,   # Limit length to stay under 100 words 
        min_new_tokens=60,    # Ensure at least 50 words 
        do_sample=True, 
        temperature=0.7,
        repetition_penalty=1.2
    )
    
    # Cleaning the output to show only the story
    full_text = story_results[0]['generated_text']
    story_content = full_text.split("<|assistant|>\n")[-1].strip()

    prefixes_to_remove = ["Picture:", "Story:", "Narrative:", "Description:"]
    for prefix in prefixes_to_remove:
        if story_content.startswith(prefix):
            story_content = story_content[len(prefix):].strip()
            
    # Cut off at the last full sentence for better readability
    if "." in story_content:
        story_content = story_content[:story_content.rindex(".")+1]
        
    return story_content
    
# --- Function 3: Text to Audio ---
def text2audio(story_text):
    _, _, audio_model = load_models()
    return audio_model(story_text)


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
                
                # Execute the 3 stages
                # Step 1: Captioning
                caption = img2text(uploaded_file)
                st.info(f"I see: {caption}")
                
                # Step 2: Story Generation
                story = text2story(caption)
                st.subheader("Your Story")
                st.write(story)
                
                # Step 3: Audio Conversion
                audio_data = text2audio(story)
                st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
                
                st.balloons()

if __name__ == "__main__":
    main()
