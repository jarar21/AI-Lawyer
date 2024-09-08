import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3

st.set_page_config(page_title="AI Law Site" ,layout="wide", initial_sidebar_state=st.session_state.get('sidebar_state', 'expanded'))
st.session_state.sidebar_state = 'expanded'

# Load environment variables from .env file
load_dotenv()

# Database setup
conn = sqlite3.connect('chat_history.db', check_same_thread=False)
cursor = conn.cursor()

# Create the chat_history table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        response TEXT NOT NULL
    )
''')
conn.commit()

# Directory containing PDF files
text_chunks_file = 'text_chunks.pkl'

@st.cache_data
def load_or_extract_text_chunks(pdf_directory):
    if os.path.exists(text_chunks_file):
        with open(text_chunks_file, 'rb') as f:
            text_chunks = pickle.load(f)
        print("Text chunks loaded from file.")
    else:
        pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]
        text_chunks = []
        for pdf_file in pdf_files:
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text:
                        paragraphs = text.split('\n\n') # Split by paragraphs
                        for paragraph in paragraphs:
                            text_chunks.append((paragraph, page_number, pdf_file))
        with open(text_chunks_file, 'wb') as f:
            pickle.dump(text_chunks, f)
        print("Text chunks extracted and saved to file.")
    return text_chunks

# Directory containing PDF files
pdf_directory = 'data'  # Update this path to where your PDFs are stored

# Load or extract text chunks
text_chunks = load_or_extract_text_chunks(pdf_directory)

# Check if text_chunks is empty
if not text_chunks:
    st.error("No text extracted from PDF files. Please check the PDF files and directory path.")
    st.stop()

# Initialize the sentence transformer model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# File to save/load embeddings
embeddings_file = 'embeddings.pkl'

@st.cache_data
def load_or_compute_embeddings(text_chunks):
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
            print("Embeddings loaded from file.")
    else:
        embeddings = embedding_model.encode([chunk[0] for chunk in text_chunks])
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print("Embeddings calculated and saved to file.")
    return embeddings

# Example usage
embeddings = load_or_compute_embeddings(text_chunks)

def search_pdfs(query, max_length=2048, similarity_threshold=0.3):
    # Encode the query
    query_embedding = embedding_model.encode([query])[0]

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]

    # Check if similarities is empty
    if not similarities:
        return None, None, None, None

    most_similar_index = similarities.index(max(similarities))
    max_similarity = similarities[most_similar_index]

    # Check if the most similar text is above the threshold
    if max_similarity < similarity_threshold:
        return None, None, None, None

    # Get the most relevant text chunk, its page number, and PDF file
    relevant_text, page_number, pdf_file = text_chunks[most_similar_index]

    # Extract heading (first line as a simple heuristic)
    heading = relevant_text.split('\n')[0]

    # Return the heading, truncated content, page number, and PDF file
    return heading, relevant_text[:max_length], page_number, pdf_file

YOUR_API_KEY = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

def generate_response(query):
    # Retrieve the most relevant text from the PDFs
    heading, context, page_number, pdf_file = search_pdfs(query)
    
    if heading is None:
        # If no relevant content is found, provide a general response
        response_content = "Query not found in the PDFs. Here's a general response: " \
                           "This topic might not be covered in the provided documents. Please consult additional resources."
        # No need to provide PDF or page information since no relevant text was found
        output = (
            f"**Content:** {response_content}\n\n"
        )
    else:
        # Define the messages for the conversation
        messages = [
            {
                "role": "system",
                "content": "You are an AI lawyer specializing in Pakistani laws. Provide accurate and detailed legal information."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}"
            },
        ]

        # Chat completion without streaming
        response = client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=messages,
        )

        response_content = response.choices[0].message.content

        # If a PDF file is found, provide details, otherwise avoid using basename on None
        pdf_info = f"**Source PDF:** {os.path.basename(pdf_file)}" if pdf_file else "**Source PDF:** Not available"
        output = (
            f"**Content:** {response_content}\n\n"
        )
    
    # Save the query and response to the database
    save_chat_history(query, response_content)

    return output

def save_chat_history(query, response):
    cursor.execute('''
        INSERT INTO chat_history (query, response)
        VALUES (?, ?)
    ''', (query, response))
    conn.commit()

def load_chat_history():
    cursor.execute('SELECT id, query, response FROM chat_history')
    return cursor.fetchall()

# Streamlit interface
st.title('AI Law Site')

st.markdown(
    """
    <style>
        /* Target elements with data-testid="column" */
        div[data-testid="column"]:nth-child(1) {
            width: calc(80% - 1rem) !important;
            flex: 1 1 calc(80% - 1rem) !important;
            min-width: calc(80% - 1rem) !important;
        }
        div[data-testid="column"]:nth-child(2) {
            width: calc(20% - 1rem) !important;
            flex: 1 1 calc(20% - 1rem) !important;
            min-width: calc(20% - 1rem) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:

    if st.button("New Chat"):
        st.session_state.sidebar_state = 'collapsed'
        st.session_state.messages = []
        st.rerun()

    if st.button("Delete Chat History"):
        if st.checkbox("Are you sure you want to delete all chat history?"):
            st.session_state.messages = []
            cursor.execute("DELETE FROM chat_history")
            conn.commit()
            st.success("Chat history deleted successfully!")
        else:
            st.warning("Please confirm deletion by checking the box.")

    # Load old chats from database
    old_chats = load_chat_history()
    if 'deletion_occurred' not in st.session_state:
        st.session_state.deletion_occurred = False
    # Display old chats in a fancy box
    st.markdown("<div style='background-color: #202222; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white;'>Pevious Chats</h2>", unsafe_allow_html=True)
    for i, chat in enumerate(old_chats):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat[1][:20] + " ...", key=f"old_chat_{i}", use_container_width=True):
                st.session_state.messages = [
                    {"role": "user", "content": chat[1]},
                    {"role": "assistant", "content": chat[2]}
                ]
                st.session_state.sidebar_state = 'collapsed'
                st.rerun()
        with col2:
            if st.button("⋮", key=f"kebab_{i}"):
                st.session_state[f"show_delete_{i}"] = not st.session_state.get(f"show_delete_{i}", False)
        
        if st.session_state.get(f"show_delete_{i}", False):
            if st.button("🗑️ Delete", key=f"delete_chat_{i}", type="primary"):
                cursor.execute('DELETE FROM chat_history WHERE id = ?', (chat[0],))
                conn.commit()
                st.session_state.deletion_occurred = True
                st.success("Chat deleted successfully!")
        if st.session_state.deletion_occurred:
            st.session_state.deletion_occurred = False
            st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Generate a response
        response = generate_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
