
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
    gen_pipe = pipeline("text-generation", model="microsoft/phi-2", device_map="cpu")
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    return img_pipe, gen_pipe, audio_pipe

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
    
    # Phi-2 responds better to a structured 'Instruction' format[cite: 1]
    prompt = f"Instruct: Write a simple 60-word bedtime story for a 5-year-old kid about {description}.\nOutput: "
    
    story_results = gen_model(
        prompt, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=50256
    )
    
    full_text = story_results[0]['generated_text']
    
    # Only capture the text AFTER 'Output: '[cite: 1]
    if "Output: " in full_text:
        story_content = full_text.split("Output: ")[-1].strip()
    else:
        story_content = full_text.replace(prompt, "").strip()
        
    # Ensure it ends with a complete sentence[cite: 1]
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
                desc = img2text(uploaded_file)
                st.info(f"I see: {desc}")
                
                story = text2story(desc)
                st.subheader("Your Story")
                st.write(story)
                
                audio_output = text2audio(story)
                st.audio(audio_output["audio"], sample_rate=audio_output["sampling_rate"])
                
                # Fun balloon animation for kids[cite: 1]
                st.balloons()


# Run the application

if __name__ == "__main__":
    main()


