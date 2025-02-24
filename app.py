#!/usr/bin/env python3
import streamlit as st
import os
import time
import tempfile

from rag import Raggy

MODELS = [
    "llama3.2:1b",
    "qwen2.5",
    "deepseek-r1:8b",
]

st.set_page_config(page_title="Raggy", initial_sidebar_state="collapsed")


def display_messages():
    st.subheader("Chat")
    for msg, is_user in st.session_state["messages"]:
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(msg)
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

        st.session_state["user_input"] = ""


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["rag_docs"]:
        print(f"{file}")
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with (
            st.session_state["ingestion_spinner"],
            st.spinner(f"Ingesting {file.name}"),
        ):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path, file.type)
            t1 = time.time()

        st.session_state["messages"].append(
            (
                f"Ingested {file.name} in {t1 - t0:.2f} seconds",
                False,
            )
        )
        os.remove(file_path)


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        try:
            st.session_state.assistant.ingest_url(url)
            st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")
        except Exception as e:
            st.error(f"Error loading document from {url}: {e}")


def update_model():
    st.session_state["assistant"] = Raggy(st.session_state.model)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = Raggy()

    with st.sidebar:
        st.divider()
        st.selectbox(
            "Select a Model",
            [model for model in MODELS],
            key="model",
            on_change=update_model,
        )

    st.header("Raggy")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=read_and_save_file,
        label_visibility="collapsed",
        key="rag_docs",
    )

    st.subheader("Read from URL")
    st.text_input(
        "🌐 Introduce a URL",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )

    # TODO: st.expander documents in db

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
