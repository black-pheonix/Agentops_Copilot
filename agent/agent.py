import json
import requests
import re
from bs4 import BeautifulSoup
from groq import Groq
from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

load_dotenv()
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("api_key"))
pc=Pinecone(api_key=os.getenv("pinecone_key"))
index = pc.Index("agentops-copilot")

MAX_ITERATIONS = 5

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search ArduPilot documentation for relevant information using semantic search. Use this first for any technical question about ArduPilot parameters, configuration, or behavior.",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The search query to find relevant documentation chunks"
                    }
                },
                "required":["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch the full content of a specific ArduPilot documentation page by URL. Use this when you already know the exact URL needed, or when search_docs returned insufficient information and you need complete page content. Do not use for general questions — use search_docs first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type": "string",
                        "description": "The full URL of the ArduPilot documentation page to fetch, for example https://ardupilot.org/dev/docs/extended-kalman-filter.html"
                    }
                },
                "required": ["url"]
            }     
            }
    }
]

def embed(texts: list) -> list:
    result = hf_client.feature_extraction(
        texts,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return result.tolist()

def search_docs(query:str, top_k:int=3) -> str:
    MIN_SCORE = 0.5
    query_embedding = embed([query])[0]
    results = index.query(
        vector = query_embedding,
        top_k = top_k,
        include_metadata = True
    )
    chunks = [match["metadata"]["text"] for match in results["matches"] if match["score"] >= MIN_SCORE]
    if not chunks:
        return "No relevant documentation found for the query."
    return "\n\n---\n\n".join(chunks)

def fetch_page(url:str) -> str:
    response = requests.get(url, timeout= 10)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "lxml")
    main = soup.find("div", itemprop="articleBody")
    if not main:
        return "Could not extract content from the page."
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r'¶', '', text)           # remove paragraph symbols
    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse 3+ newlines into 2
    text = text.strip()
    return text[:3000]

def run_tool(tool_name:str, tool_args:dict) -> str:
    if tool_name == "search_docs":
        return search_docs(**tool_args)
    elif tool_name == "fetch_page":
        return fetch_page(**tool_args)
    else:
        return f"Unknown tool: {tool_name}"    

def run_agent(user_question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an ArduPilot documentation assistant. Use the search_docs tool first for technical questions. If the search results are insufficient, use the fetch_page tool with a relevant ardupilot.org URL. Give direct answers once you have enough information."
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration+1}")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                }
            )

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"Calling tool: {tool_name} with args: {tool_args}")

                result = run_tool(tool_name, tool_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                )
        elif finish_reason == "stop":
            return message.content
    return "Agent reached maximum iterations without a final answer."

if __name__ == "__main__":
    while True:
        print("You:")
        question = input()
        if question.lower() == "exit":
            break
        answer = run_agent(question)
        print(f"\nAgent: {answer}")
