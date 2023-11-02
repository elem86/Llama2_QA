# Import necessary libraries
import os
import transformers
import pinecone
import streamlit as st

from torch import cuda, bfloat16
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set up the embedding model ID and device (GPU if available, otherwise CPU)
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

# Initialize the Hugging Face Embedding Pipeline
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={"device": device},
    encode_kwargs={"device": device, "batch_size": 32},
)

# Initialize Pinecone
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY")
    or "PINECONE_API_KEY", # The actual PINECONE_API_KEY 
    environment=os.environ.get("PINECONE_ENVIRONMENT") or "gcp-starter",
)
index = pinecone.Index("llama-2-rag")

# Set up the Hugging Face model for text generation
model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

hf_auth = "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"
model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_auth)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=hf_auth,
)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    temperature=0.7, # You can play with this value to see how you like the different results.
    max_new_tokens=512,
    repetition_penalty=1.1,
)

# Set up the LangChain pipeline for question-answering
llm = HuggingFacePipeline(pipeline=generate_text)
text_field = "text"
vectorstore = Pinecone(index, embed_model.embed_query, text_field)
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
)


# Define the Streamlit interface for interactive testing
def interactive_testing():
    st.title("Interactive Testing with Llama Model")
    user_input = st.text_input("You: ")

    if user_input:
        answer = rag_pipeline({"query": user_input})
        st.write(f"Answer: {answer['result']}")


# Start the Streamlit app
interactive_testing()
