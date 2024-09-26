from transformers import DPRQuestionEncoder, DPRContextEncoder, RagTokenizer
import faiss
import torch

# Initialize models and retrievers
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
device = torch.device("cpu")  # Force to CPU for now

# FAISS index setup
dimension = 768
index = faiss.IndexFlatL2(dimension)

# Sample document list
documents = [{"id": "doc1", "text": "This is a finance document."}]

def index_documents(docs):
    for doc in docs:
        inputs = tokenizer(doc['text'], return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            doc_embeddings = context_encoder(**inputs).pooler_output.detach().cpu().numpy()
        index.add(doc_embeddings)

def search_query(query, top_k=5):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output.detach().cpu().numpy()

    D, I = index.search(question_embedding, k=top_k)
    return [{"doc_id": documents[i]['id'], "score": float(D[0][j])} for j, i in enumerate(I[0])]

# Test
index_documents(documents)
result = search_query("finance")
print(result)

