import streamlit as st
import time
import random
from openai import OpenAI
import os
import ollama
import openai


def ollama_model(prompt, model="llama3"):
    stream_response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )
    for chunk in stream_response:
        yield chunk["message"]["content"]


def openai_model(prompt, model="gpt-4o"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
            "I don't have real time access to data. I am only an AI",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def main():
    st.title("El Chato/GPT-4o")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    model_option = st.sidebar.selectbox(
        "Choose the model",
        ["OpenAI", "llama3", "gemma", "phi3"],
    )

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            if model_option == "OpenAI":
                stream = openai_model(prompt, model=st.session_state["openai_model"])
            elif model_option == "llama3":
                stream = ollama_model(prompt, model="llama3")
            elif model_option == "gemma":
                stream = ollama_model(prompt, model="gemma:latest")
            elif model_option == "phi3":
                stream = openai_model(prompt, model="phi3")

            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
