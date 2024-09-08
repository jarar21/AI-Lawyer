import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import streamlit as st
from openai import OpenAI

# Directory containing PDF files
pdf_directory = 'data'  # Update this path to where your PDFs are stored

# List all PDF files in the directory
pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

# Extract text from all PDFs
text_chunks = []
for pdf_file in pdf_files:
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                paragraphs = text.split('\n\n')  # Split by paragraphs
                for paragraph in paragraphs:
                    text_chunks.append((paragraph, page_number, pdf_file))

# Check if text_chunks is empty
if not text_chunks:
    st.error("No text extracted from PDF files. Please check the PDF files and directory path.")
    st.stop()

# Initialize the sentence transformer model
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
        output = response_content
    else:
        # Define the messages for the conversation
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an artificial intelligence assistant and you need to "
                    "engage in a helpful, detailed, polite conversation with a user."
                ),
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

        # Access the content correctly
        response_content = response.choices[0].message.content
        save_chat_history(query, response_content)  # Call with two arguments

        # Format the output with citations
        output = (
            f"**Heading:** {heading}\n\n"
            f"**Content:** {response_content}\n\n"
            f"**Page Number:** {page_number}\n"
            f"**Source PDF:** {os.path.basename(pdf_file)}"
        )
        
    return output

def save_chat_history(query, response_content):
    with open("responses.txt", "a") as file:  # Use 'a' to append to the file
        file.write(f"Query: {query}\n")
        file.write(f"Response:\n{response_content}\n")
        file.write("="*50 + "\n")

def load_chat_history():
    chat_history = []
    if os.path.exists("responses.txt"):
        with open("responses.txt", "r") as file:
            lines = file.readlines()
            current_chat = {"query": "", "response": ""}
            response_lines = []
            for line in lines:
                line = line.strip()  # Remove any leading/trailing whitespace
                if line.startswith("Query:"):
                    # If there's an existing query, append it to the history
                    if current_chat["query"]:
                        current_chat["response"] = "\n".join(response_lines)
                        chat_history.append(current_chat)
                    # Start a new chat entry
                    current_chat = {"query": line.replace("Query:", "").strip(), "response": ""}
                    response_lines = []
                elif line.startswith("="*50):
                    # End of a chat entry, append it to the history
                    if current_chat["query"]:
                        current_chat["response"] = "\n".join(response_lines)
                        chat_history.append(current_chat)
                    current_chat = {"query": "", "response": ""}
                    response_lines = []
                else:
                    # Collect response lines
                    response_lines.append(line)
            # Append the last entry if it exists
            if current_chat["query"]:
                current_chat["response"] = "\n".join(response_lines)
                chat_history.append(current_chat)
    return chat_history

# Streamlit interface
st.title('LAIER LAW SITE')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])
# Load old chats
old_chats = load_chat_history()

# Display old chats in a fancy box
st.sidebar.markdown("<div style='background-color: #202222; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color: white;'>Old Chats</h4>", unsafe_allow_html=True)
for i, chat in enumerate(old_chats):
    if st.sidebar.button(chat["query"], key=f"old_chat_{i}"):
        st.session_state.messages = [
            {"role": "user", "content": chat["query"]},
            {"role": "assistant", "content": chat["response"]}
        ]
st.sidebar.markdown("</div>", unsafe_allow_html=True)

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

    # Generate a response
    response = generate_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})