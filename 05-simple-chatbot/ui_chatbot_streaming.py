import streamlit as st
from chatbot import workflow
from langchain_core.messages import HumanMessage


config={"configurable": {"thread_id": '1'}}
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for messages in st.session_state["message_history"]:
        with st.chat_message("role"):
            st.markdown(messages['content'])

user_input = st.chat_input("Type here...")

if user_input:
    st.session_state["message_history"].append({"role":"user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        ai_msg = st.write_stream(message_chunk.content for message_chunk, metadata in workflow.stream({'messages': [HumanMessage(content=user_input)]}, config=config, stream_mode='messages'))
    st.session_state["message_history"].append({"role":"assistant", "content": ai_msg})
