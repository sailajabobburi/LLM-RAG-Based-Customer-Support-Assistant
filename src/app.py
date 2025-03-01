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
        .stApp {
            margin-top: 120px;
        }

        header[data-testid="stHeader"] {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 100px;
            background: #F5F7FA;
            z-index: 999990;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 20px;
            flex-direction: column;
        }

        header[data-testid="stHeader"] img {
            height: 50px;
            margin-bottom: 5px;
        }

        header[data-testid="stHeader"] h1 {
            color: #004AAD;
            font-size: 22px;
            font-weight: bold;
            margin: 0;
            text-align: center;
        }

        /* Chat container */
        .chat-container {
            border-radius: 12px;
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .user-message {
            background-color: #0078FF;
            color: white;
            padding: 10px;
            border-radius: 12px;
            margin: 5px 10px;
            text-align: right;
            align-self: flex-end;
            max-width: 75%;
        }

        .bot-message {
            background-color: #F1F1F1;
            color: black;
            padding: 10px;
            border-radius: 12px;
            margin: 5px 10px;
            text-align: left;
            align-self: flex-start;
            max-width: 75%;
        }

        /* Processing message */
        .processing-message {
            color: #0078FF;
            font-size: 14px;
            font-weight: bold;
            margin-top: 5px;
            text-align: center;
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

# ---- RESET CHAT HISTORY ON REFRESH ----
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [("Lisa", "Hi, I'm Lisa, your AI-assisted assistant! How can I help you today?")]

# ---- CHAT CONTAINER (DYNAMIC) ----
chat_container = st.container()
processing_placeholder = st.empty()  # Placeholder for processing message

# Function to update chat UI
def update_chat():
    chat_container.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, message in st.session_state["chat_history"]:
        chat_container.markdown(
            f'<div class="{"user-message" if role == "User" else "bot-message"}">{message}</div>',
            unsafe_allow_html=True
        )
    chat_container.markdown('</div>', unsafe_allow_html=True)

# Initial chat display
update_chat()

# ---- INPUT AREA & ASK BUTTON ----
user_text = st.text_input("Type your message...", key="user_input", placeholder="Type your message...")
ask_button = st.button("Ask")  # "Ask" button

def send_message():
    if user_text:  # Only process if user input exists
        # Append user message to chat history
        st.session_state.chat_history.append(("User", user_text))

        # Show processing message below input field
        processing_placeholder.markdown('<div class="processing-message">Processing... Please wait</div>', unsafe_allow_html=True)

        # Update chat UI before processing
        update_chat()

        # Prepare conversation context for LLM
        full_conversation = "\n".join(
            [f"{role}: {msg}" for role, msg in st.session_state["chat_history"]]
        )

        # Load vector DB and data
        index = faiss.read_index(config.VECTOR_STORE_FILE)
        df = pd.read_csv(config.INPUT_DATA_PATH)

        # Get similar responses
        distances, indices = semantic_similarity(user_text, index, model=config.EMBEDDING_MODEL)
        top_similar_instructions = df.iloc[indices[0]].reset_index(drop=True)
        top_similar_instructions['response'] = distances[0]

        # Generate AI response
        llm_response = generate_response(user_text, top_similar_instructions['response'].tolist(), full_conversation)

        # Store bot response in chat history
        st.session_state.chat_history.append(("Lisa", llm_response))

        # Update chat UI again after response
        update_chat()

        # Remove processing message
        processing_placeholder.empty()

        # **Fix:** Reset input field correctly by removing key instead of modifying
        st.session_state.pop("user_input", None)  # Removes 'user_input' key safely

# ---- CALL FUNCTION WHEN BUTTON IS CLICKED ----
if ask_button:
    send_message()

st.markdown('</div>', unsafe_allow_html=True)
