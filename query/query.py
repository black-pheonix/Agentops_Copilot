from groq import Groq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("pinecone_key"))
index = pc.Index("agentops-copilot")
client = Groq(api_key=os.getenv("api_key"))

messages = [{
    "role": "system", "content": """You are an Ardupilot documentation assistant.
                Answer the questions using ONLY the context provided.
                If the context does not contain enough information, say so clearly.
                Do not make up information"""
}]

def query_rag(question: str, top_k: int = 3) -> str:
    query_embedding = embedder.encode(question).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata= True
    )
    # print("--- RETRIEVED CHUNKS ---")
    for i, match in enumerate(results["matches"]):
        print(f"\nChunk {i} | Score: {match['score']:.4f}")
        # print(match["metadata"]["text"])
    # print("--- END CHUNKS ---\n")
    MIN_SCORE = 0.7
    context_chunks = [match["metadata"]["text"] for match in results["matches"] if match["score"] >= MIN_SCORE]
    context = "\n\n---\n\n".join(context_chunks)
    temp_messages = messages + [{
        "role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"
    }]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages = temp_messages
    )
    messages.append({
        "role": "user", "content": question
    })
    messages.append({
        "role": "assistant", "content": response.choices[0].message.content
    })
    return response.choices[0].message.content

while True:
    print("You: ")
    question = input()
    if question.lower() == "exit":
        break
    answer = query_rag(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")