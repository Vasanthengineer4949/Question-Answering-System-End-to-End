import transformers
from sentence_transformers import SentenceTransformer
model = transformers.AutoModel.from_pretrained('bert-base-uncased-firstqa-squad_v2')
model.push_to_hub(repo_url = "Vasanth/bert-base-uncased-qa-squad2")
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1-qa-squad2-retriever')
model.save_to_hub(repo_name="multi-qa-MiniLM-L6-cos-v1-qa-squad2-retriever")