# Llama QA Model with Pinecone for the DataSpeak project

This project is an adaptation of the Llama QA model showcased in the Pinecone examples. It leverages the power of Pinecone's vector search capabilities combined with Hugging Face's transformers to create a question-answering system.

## Features

- Uses the Llama model for text generation.
- Integrates with Pinecone for efficient vector search.
- Provides an interactive interface using Streamlit.

## Setup

### 0. Getting the necessary API keys:

The code was built on/re-built based on the [Hugging Face article](https://agi-sphere.com/retrieval-augmented-generation-llama2/). In this article, it is explained that you will need to get a pinecone API key, a pinecone environment, Hugging Face authorization token. On top of these for my solution you will also need a ngrok API key.

These keys can be acquired relatively fast, within minutes so it should not be an issue.

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
I would recommend loading the notebook file into Google Collab where you have the GPU power. Once you upload the app to the collab notebook you can run the notebook itself. When the Streamlit app is running, you can interact with the Llama QA model by entering questions into the provided input field. The model will retrieve relevant answers based on the underlying dataset and the capabilities of the Llama model.


## DEMO

![image](https://github.com/elem86/Llama2_QA/assets/48759327/234bae2a-aa70-44c4-a768-8fab17e93b8e)

![image](https://github.com/elem86/Llama2_QA/assets/48759327/82fdaedc-5bc5-46cf-b7ac-cfa3623a3b18)

![image](https://github.com/elem86/Llama2_QA/assets/48759327/4b803bbc-4c8b-4fff-bdea-6282089c0426)

![image](https://github.com/elem86/Llama2_QA/assets/48759327/db25312f-8884-44c3-8296-ddb9421f97a8)

![image](https://github.com/elem86/Llama2_QA/assets/48759327/e03cdfbe-8930-493d-a342-66c04ece1c82)

![image](https://github.com/elem86/Llama2_QA/assets/48759327/2b5aaa3c-0571-4e4e-85bd-20c974146f37)





