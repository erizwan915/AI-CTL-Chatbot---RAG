from ollama import chat
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

def load_expanded_chunks(file_path="expanded_tutor_chunks.csv"):
    df = pd.read_csv(file_path)
    return df["Chunk"].dropna().tolist()

def build_index(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, chunks, model

def get_context(question, index, model, chunks, embeddings, top_k=5):
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), top_k)
    context = "\n".join(chunks[i] for i in I[0])
    return context, D[0].tolist()  # <-- distances as list


def chatbot(user_message, messages, index, chunks, embeddings, embed_model):
    context, distances = get_context(user_message, index, embed_model, chunks, embeddings)

    messages.append({
        "role": "user",
        "content": f"Use this context to answer:\n{context}\n\nQuestion: {user_message}"
    })

    response = chat(model="llama3.2", messages=messages)
    reply = response["message"]["content"]

    messages.append({"role": "assistant", "content": reply})
    return reply, distances, context

