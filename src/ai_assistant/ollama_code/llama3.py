import ollama


def ollama_model(model="llama3"):
    stream = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Why is the sky blue?.",
            }
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    ollama_model(model="llama3")
    print("Done!")
