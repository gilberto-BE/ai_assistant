import streamlit as st
import time
import random


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
    st.title("El Chato")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How are you today?"):

        with st.chat_message("User"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "User", "content": prompt})

    response = f"El chato: {prompt}"
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
        # st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
