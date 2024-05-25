import streamlit as st


def main():
    st.title("Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("How are you today?"):

        with st.chat_message("User"):
            st.markdown(prompt)
        st.session_state.messages.append({'role': "User", 'content': prompt})



if __name__ == "__main__":
    main()