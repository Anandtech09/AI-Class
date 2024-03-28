import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("2003achu/ai_class")
model = AutoModelForSeq2SeqLM.from_pretrained("2003achu/ai_class")

# Function to generate AI poem
def generate(prompt):
    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_length=150, num_return_sequences=3, temperature=0.9)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]

# Class description
st.sidebar.title('About')
st.sidebar.info(
    "This Class Name Generator helps you generate Python class names based on the provided description."
    "\n\nFor example, if you need a class to extract text and metadata from PDF documents, you can input "
    "a description like 'Extracts text and metadata from PDF documents for further processing' and get a "
    "suggested class name like 'PDFExtractor'."
)

# Main content
st.title('Class Name Generator')
input_description = st.text_area("Enter class description here", height=100)

# Generate class name upon button click
if st.button("Generate Class Name"):
    if input_description:
        # Display spinner while generating class name
        with st.spinner("Generating class name..."):
            class_name = generate(input_description)
            st.text(f"Suggested Class Name: {class_name}")

# Footer
st.markdown("---")
st.markdown("This app helps you generate Python class names based on the provided description.")