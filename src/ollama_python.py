# import ollama
# if __name__ == "__main__":
#     stream = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': 'why is the sky blue?',},], stream=True)

#     for chunk in stream:
#         print(chunk['message']['content'], end='',flush=True)

import dataclass
from openai import OpenAI


@dataclass
class Message:
    role: str
    content: str


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)

response = client.chat.completions.create(
    model="llama3.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The LA Dodgers won in 2020."},
        # {"role": "user", "content": "Where was it played?"}
    ],
)
print(response.choices[0].message.content)
