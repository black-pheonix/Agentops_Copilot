from groq import Groq
from dotenv import load_dotenv
from anthropic import Anthropic
import os

load_dotenv()

Groq_key = os.getenv("api_key")
# claude_client = Anthropic(api_key=os.getenv("claude_key"))

client = Groq(api_key=Groq_key)

# messages=[
#         {"role": "system", "content": "You are a UAV expert. Answer consisely and technically."},
#         {"role": "user", "content": "What is a flight controller?"}
#     ]
# response = claude_client.messages.create(
#     model = "claude-haiku-4-5-20251001",
#     max_tokens = 1024,
#     system=messages[0]["content"],
#     messages=messages[1:],

# )

response = client.chat.completions.create(
    model= "llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a UAV expert. Answer consisely and technically."},
        {"role": "user", "content": "What is a flight controller?"}
    ],
    response_format={"type": "json_object"}
)
print(response)
