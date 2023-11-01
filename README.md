# Llama QA Model with Pinecone for the DataSpeak project

This project is an adaptation of the Llama QA model showcased in the Pinecone examples. It leverages the power of Pinecone's vector search capabilities combined with Hugging Face's transformers to create a question-answering system.

## Features

- Uses the Llama model for text generation.
- Integrates with Pinecone for efficient vector search.
- Provides an interactive interface using Streamlit.

## Setup

### 1. Clone the Repository:

```bash
git clone https://github.com/elem86/Llama2_QA.git
```

### 2. Install Dependencies:
Ensure you have Python 3.7 or later installed.

```bash
pip install -r requirements.txt
```

### 3. Set Up Pinecone:

- Sign up for a Pinecone account if you haven't already.
- Retrieve your Pinecone API key and set it as an environment variable or replace the placeholder in the code.

### 4. Run the app from the provided Dataspeak_ML.ipynb file:

```python
# Set up ngrok
ngrok.set_auth_token('YOUR_NGROK_TOKEN')  # Replace 'YOUR_NGROK_TOKEN' with your token

# Start Streamlit app
os.system("streamlit run llama2.py &")

# Get ngrok public URL
public_url = ngrok.connect(port=8501)
print("Streamlit UI can be accessed on:", public_url)
```


## Usage
Once the Streamlit app is running, you can interact with the Llama QA model by entering questions into the provided input field. The model will retrieve relevant answers based on the underlying dataset and the capabilities of the Llama model.
