import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Page config
st.set_page_config(page_title="Robotics Forum Chatbot", page_icon="🤖")

# Cache model loading
@st.cache_resource
def load_model():
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load fine-tuned weights
    model = PeftModel.from_pretrained(model, "samadhanzanzurne/robotics-chatbot")
    
    return tokenizer, model

def generate_response(question, tokenizer, model):
    prompt = f"""### Instruction:
{question}

### Input:


### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("### Response:")[-1].strip()
    
    return response

# UI
st.title("🤖 The Robotics Forum Chatbot")
st.markdown("Ask me anything about our club!")

# Load model
with st.spinner("Loading chatbot model..."):
    tokenizer, model = load_model()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, tokenizer, model)
            st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This chatbot is fine-tuned specifically for The Robotics Forum club.")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
