import streamlit as st
import faiss
import pandas as pd
import config
from helper import semantic_similarity, generate_response

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Assisted Customer Support", page_icon="💡", layout="wide")

# ---- CUSTOM CSS FOR FIXED HEADER & CHAT UI ----
st.markdown(
    """
    <style>
        /* Fix header, make it fully visible and static */
        body {
            margin-top: 120px; /* Adjusted to accommodate fixed header */
        }

        .stApp header[data-testid="stHeader"] {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 120px;
            background: #F5F7FA;
            z-index: 999990;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 20px;
            flex-direction: column;
        }

        .stApp header[data-testid="stHeader"] img {
            height: 50px;
            margin-bottom: 5px;
        }

        .stApp header[data-testid="stHeader"] h1 {
            color: #004AAD;
            font-size: 24px;
            font-weight: 700;
            margin: 0;
            text-align: center;
        }

        /* Main content layout */
        .main-content {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
        }

        /* Chat container */
        .chat-container {
            background: none;
            border-radius: 12px;
            height: calc(100vh - 280px); /* Adjust height to fit screen */
            overflow-y: auto; /* Add a scrollbar when content overflows */
            margin-bottom: 10px;
        }

        .user-message {
            background-color: #0078FF;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: right;
            width: fit-content;
            max-width: 80%;
            margin-left: auto;
        }

        .bot-message {
            background-color: #F1F1F1;
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: left;
            width: fit-content;
            max-width: 80%;
        }

        /* Input area */
        .input-area {
            position: relative;
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }

        .input-area input[type="text"] {
            width: 100%;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 8px;
            font-size: 16px;
        }

        /* Processing message */
        .processing-message {
            text-align: center;
            font-style: italic;
            color: #888;
        }

        /* Hide default Streamlit header */
        .stApp > header {
            display: none !important;
        }

        /* Ensure content starts below our custom header */
        .stApp > .main {
            margin-top: 120px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- FIXED HEADER (STATIC LOGO & TITLE) ----
st.markdown(
    """
    <header data-testid="stHeader">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg">
        <h1>AI Assisted Customer Support</h1>
    </header>
    """,
    unsafe_allow_html=True
)

# ---- MAIN CONTENT ----
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ---- CHAT MEMORY ----
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [("Lisa", "Hi, I'm Lisa, your AI-assisted assistant! How can I help you today?")]

# ---- DISPLAY CHAT MESSAGES ----
# Create a chat container with a scrollbar
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, message in st.session_state["chat_history"]:
    st.markdown(f'<div class="{"user-message" if role == "User" else "bot-message"}">{message}</div>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ---- INPUT AREA ----
def send_message():
    if st.session_state.user_input and not st.session_state.get("processing", False):
        st.session_state.processing = True

        st.session_state.chat_history.append(("User", st.session_state.user_input))
        text = st.session_state.user_input

        full_conversation = "\n".join(
            [f"{role}: {msg}" for role, msg in st.session_state["chat_history"]]
        )

        index = faiss.read_index(config.VECTOR_STORE_FILE)
        df = pd.read_csv(config.INPUT_DATA_PATH)

        distances, indices = semantic_similarity(text, index, model=config.EMBEDDING_MODEL)
        top_similar_instructions = df.iloc[indices[0]].reset_index(drop=True)
        top_similar_instructions['response'] = distances[0]

        llm_response = generate_response(text, top_similar_instructions['response'].tolist(), full_conversation)

        st.session_state.chat_history.append(("Lisa", llm_response))
        st.session_state.user_input = ""
        st.session_state.processing = False


# Initialize processing state
if "processing" not in st.session_state:
    st.session_state.processing = False

# Text input and send button
user_input = st.text_input("Type your message...", key="user_input", disabled=st.session_state.processing,
                           on_change=send_message)
send_button = st.button("Send", on_click=send_message, disabled=st.session_state.processing)

# Display processing message
if st.session_state.processing:
    st.markdown('<div class="processing-message">Processing... Please wait.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
