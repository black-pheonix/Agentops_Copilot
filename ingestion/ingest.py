import requests
from bs4 import BeautifulSoup
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("pinecone_key"))
index = pc.Index("agentops-copilot")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])

urls = ["https://ardupilot.org/copter/docs/common-apm-navigation-extended-kalman-filter-overview.html",
        "https://ardupilot.org/dev/docs/extended-kalman-filter.html"
        ]

index.delete(delete_all=True)

for url in urls:
    url_hash = hashlib.md5(url.encode()).hexdigest()
    response = requests.get(url, timeout=10)
    response.encoding = "utf-8"
    html = response.text
    soup = BeautifulSoup(html, "lxml")

    for tag in soup([
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "noscript",
        "iframe",
        "svg",
        "form"
    ]):
        tag.decompose()

    main = soup.find("div", itemprop="articleBody")
    if not main:
        raise ValueError("Could not find main content — check selector")
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r'¶', '', text)           # remove paragraph symbols
    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse 3+ newlines into 2
    text = text.strip()

    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if len(c) > 100] 

    structured_chunks = []

    for i, chunk in enumerate(chunks):
        structured_chunks.append(
            {
                "text": chunk,
                "metadata": {
                    "source_url": url,
                    "chunk_index": i,
                    "scraped_at":datetime.now().isoformat(),
                    "page_title": soup.title.string if soup.title else "Unknown",
                }
            }
        )

    texts = [c["text"] for c in structured_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    vectors = []

    for i, (chunk, embedding) in enumerate(zip(structured_chunks, embeddings)):
        vectors.append(
            {
                "id": f"{url_hash}-chunk-{i}",
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
            }
        )
    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} vectors")
print(index.describe_index_stats())
