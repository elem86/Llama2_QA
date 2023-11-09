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
    or "PINECONE_API_KEY", # Your PINECONE_API_KEY 
    environment=os.environ.get("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT", # Your PINECONE_ENVIRONMENT 
)
index = pinecone.Index("llama-2-rag")

# This section below is where first you need to load the dataset into the Pinecone index. You should only use it once, otherwise, it slows down the process.

#Load the data
# data = pd.read_csv("/content/drive/MyDrive/DS/Example.csv")

# # Embed the documents and index them in Pinecone
# batch_size = 32
# total_batches = len(data) // batch_size + (len(data) % batch_size != 0)
# progress = st.progress(0)  # Initialize the progress bar in Streamlit

# for i in tqdm(range(0, len(data), batch_size)):
#     i_end = min(len(data), i + batch_size)
#     batch = data.iloc[i:i_end]
#     ids = [f"{x['Id_question']}" for i, x in batch.iterrows()]
#     texts = [x["Context"] for i, x in batch.iterrows()]
#     embeds = embed_model.embed_documents(texts)

#     # Store essential metadata for each document(this is an example below based on my own data)
#     metadata = [
#         {
#             "id": x["Id_question"],
#             "title": x["Cleaned_Title"],
#             "answer": x["Cleaned_Body_answer"][
#                 :100
#             ],  # Store only the first 100 characters
#             "score": x["Score_answer"],
#         }
#         for i, x in batch.iterrows()
#     ]

#     # Update the Pinecone index with the new embeddings and metadata
#     index.upsert(vectors=zip(ids, embeds, metadata))

#     # Update the Streamlit progress bar
#     progress_value = min((i + batch_size) / len(data), 1.0)
#     progress.progress(progress_value)

# Set up the Hugging Face model for text generation
model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

hf_auth = "HF_AUTH_TOKEN" # Your HuggingFace token
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
    temperature=0.1, # You can play with this value to see how you like the different results.
    max_new_tokens=512,
    repetition_penalty=1.1,
)

# Set up the LangChain pipeline for question-answering
llm = HuggingFacePipeline(pipeline=generate_text)
text_field = "text"
vectorstore = Pinecone(index, embed_model.embed_query, text_field)
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.7},
    ),
    chain_type="stuff",
)

def interactive_testing():
    st.title("Interactive Testing with Llama Model")
    user_input = st.text_input("You: ")

    if user_input:
        response = rag_pipeline({"query": user_input})
        answer = response['result']

        # Check if the last sentence is a question and remove it
        if answer.endswith('?'):
            answer = ' '.join(answer.split('?')[:-1]) + '.'

        st.write(f"Answer: {answer}")

# Start the Streamlit app
interactive_testing()
