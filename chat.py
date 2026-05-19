from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("api_key")

client = Groq(api_key = api_key)


messages = [
    {"role": "system", "content": "You are a UAV expert. Answer consisely and technically."},
]

while True:
    print("The CHATBOT")
    print("You:")
    user_input = input()
    if "exit" in user_input.lower():
        break
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages = messages
    )
    response_message = response.choices[0].message.content
    print(response_message)
    messages.append({"role":"assistant", "content": response_message})
print (messages)
