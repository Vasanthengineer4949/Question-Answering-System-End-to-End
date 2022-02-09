from xml.etree.ElementInclude import include
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1-qa-squad2-retriever')

pinecone.init(
    api_key='2defc39d-ff30-4384-8772-9965d1b7e931',
    environment='us-west1-gcp'
)

index = pinecone.Index('firstqa-index')

st.write("""

# Question Answering System - Squad_V2  with Inferencing API from HuggingFace

A simple QA system that uses retriever and reader model connected to a Vector Database from pinecone. 

Ask me a question and I'll try to answer it.

""")

que = st.text_input("Ask me a question", '')

if que != "":
    
    xq = model.encode([que]).tolist()
    xc = index.query(xq, top_k=5, include_metadata=True)
    for context in xc['results'][0]['matches']:
        retrieve_out = context['metadata']['text']

    import requests

    API_URL = "https://api-inference.huggingface.co/models/Vasanth/bert-base-uncased-qa-squad2"
    API_TOKEN = "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"

    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": {
            "question": que,
            "context": retrieve_out,
        },
    })

    print(output)
    # model_name = "bert-base-uncased-firstqa-squad_v2"
    # question_answerer = pipeline("question-answering", model=model_name, tokenizer=model_name)
    # answer = question_answerer(
    # question= query,
    # context= retrieve_out
    # )
    st.write(output["answer"])

    