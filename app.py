import os
import psycopg2
import numpy as np
from openai import AzureOpenAI
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
import tempfile
import csv
import speech_recognition as sr
import pyttsx3
import time
from datetime import timedelta

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")  # 2025-01-01-preview
)

# Initialize pyttsx3 engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Use the first available voice

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

# Function to get embeddings
def get_embedding(text, model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")):
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)

# Retrieve data from marvel_characters_phase_three
def retrieve_phase_three_documents(query, top_k=3):
    query_embedding = get_embedding(query)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT character_name, real_name, actor, powers, debut_movie, debut_year, team_affiliation, status, embedding
                FROM marvel_characters_phase_three
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """, (query_embedding.tolist(), top_k))
            results = cur.fetchall()
            print(f"Retrieved documents: {results}")  # Debug
        return [f"Name: {row[0]}, Real Name: {row[1]}, Actor: {row[2]}, Powers: {row[3]}, Debut Movie: {row[4]}, Debut Year: {row[5]}, Team: {row[6]}, Status: {row[7]}" for row in results]
    finally:
        conn.close()

# Function to retrieve documents from a specified table
def retrieve_file_documents(query, table_name, top_k=2):
    query_embedding = get_embedding(query)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT file_name, content
                FROM {table_name}
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """, (query_embedding.tolist(), top_k))
            results = cur.fetchall()
            print(f"Retrieved file documents from {table_name}: {results}")  # Debug
        return [f"File: {row[0]}, Content: {row[1]}" for row in results]
    finally:
        conn.close()

# Process uploaded files (PDF and CSV)
def process_uploaded_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        file_path = tmp_file.name
    
    if file_extension == "pdf":
        process_pdf_file(file_path, uploaded_file.name)
    elif file_extension == "csv":
        process_csv_file(file_path, uploaded_file.name)
    
    os.remove(file_path)

def process_pdf_file(file_path, file_name):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            with pdfplumber.open(file_path) as pdf:
                content = "".join(page.extract_text() or "" for page in pdf.pages)
            if not content:
                content = "No extractable text found in PDF."
            embedding = get_embedding(content)
            cur.execute("SELECT file_name FROM marvel_characters_phase_three_documents WHERE file_name = %s", (file_name,))
            exists = cur.fetchone()
            if exists:
                cur.execute("""
                    UPDATE marvel_characters_phase_three_documents
                    SET content = %s, embedding = %s::vector
                    WHERE file_name = %s
                """, (content, embedding.tolist(), file_name))
            else:
                cur.execute("""
                    INSERT INTO marvel_characters_phase_three_documents (file_name, content, embedding)
                    VALUES (%s, %s, %s::vector)
                """, (file_name, content, embedding.tolist()))
            conn.commit()
            st.success(f"PDF {file_name} processed and stored.")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        conn.rollback()
    finally:
        conn.close()

def process_csv_file(file_path, file_name):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                content = "".join(line for line in csvfile)
            if not content:
                content = "No extractable text found in CSV."
            embedding = get_embedding(content)
            cur.execute("SELECT file_name FROM marvel_characters_phase_three_documents WHERE file_name = %s", (file_name,))
            exists = cur.fetchone()
            if exists:
                cur.execute("""
                    UPDATE marvel_characters_phase_three_documents
                    SET content = %s, embedding = %s::vector
                    WHERE file_name = %s
                """, (content, embedding.tolist(), file_name))
            else:
                cur.execute("""
                    INSERT INTO marvel_characters_phase_three_documents (file_name, content, embedding)
                    VALUES (%s, %s, %s::vector)
                """, (file_name, content, embedding.tolist()))
            conn.commit()
            st.success(f"CSV {file_name} processed and stored.")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        conn.rollback()
    finally:
        conn.close()

# Generate response
def generate_response(retrieved_docs, chat_history):
    context = "\n".join(retrieved_docs)
    
    system_message = {
        "role": "system",
        "content": f"""You are a Marvel expert chatbot. Use the following Marvel character information to answer the user's query.
Character Information:
{context}

You have access to the conversation history. Use it to answer follow-up questions.
"""
    }

    messages = [system_message] + chat_history

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT"),
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Get character names for selection
@st.cache_data
def get_character_names():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT character_name FROM marvel_characters_phase_three WHERE character_name IS NOT NULL")
            names = [row[0] for row in cur.fetchall()]
            print(f"Retrieved character names: {names}")  # Debug
            return names
    finally:
        conn.close()

# Voice input function
def take_command():
    listener = sr.Recognizer()
    with sr.Microphone() as source:
        print('listening...')
        st.info("Listening...")
        try:
            voice = listener.listen(source, timeout=5, phrase_time_limit=5)
            command = listener.recognize_google(voice).lower()
            if 'jarvis' in command:
                command = command.replace('jarvis', '')
            return command
        except sr.UnknownValueError:
            st.warning("Sorry, I couldn't understand the audio.")
            return None
        except sr.RequestError:
            st.error("Error with the speech recognition service.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

# Voice output function
def talk(text):
    try:
        engine.endLoop()
    except:
        pass
    engine.say(text)
    engine.runAndWait()

# Character comparison
def compare_characters(char1, char2):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT powers, debut_year
                FROM marvel_characters_phase_three
                WHERE character_name = %s
                LIMIT 1
            """, (char1,))
            result1 = cur.fetchone()
            print(f"Query 1 result for {char1}: {result1}")  # Debug
            cur.execute("""
                SELECT powers, debut_year
                FROM marvel_characters_phase_three
                WHERE character_name = %s
                LIMIT 1
            """, (char2,))
            result2 = cur.fetchone()
            print(f"Query 2 result for {char2}: {result2}")  # Debug
        if result1 and result2:
            powers1, year1 = result1
            powers2, year2 = result2
            return f"Comparing {char1} and {char2}:\n- Powers: {powers1} vs {powers2}\n- Debut Year: {year1} vs {year2}"
        return "Characters not found or data missing."
    finally:
        conn.close()

    # Generate quiz question
def generate_quiz_question():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT character_name, debut_movie FROM marvel_characters_phase_three ORDER BY RANDOM() LIMIT 1")
            result = cur.fetchone()
            print(f"Quiz question data: {result}")  # Debug
            if result:
                name, movie = result
                return f"What movie did {name} debut in?", movie
            return "No data available for quiz.", None
    finally:
        conn.close()

# Store chat history
def store_chat_history(role, content):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO marvel_characters_phase_three_history (role, content, timestamp)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
            """, (role, content))
            conn.commit()
    except Exception as e:
        print(f"Error storing chat history: {e}")
        conn.rollback()
    finally:
        conn.close()

# Retrieve chat conversations
def retrieve_chat_conversations():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, role, content, timestamp
                FROM marvel_characters_phase_three_history
                ORDER BY timestamp ASC
            """)
            messages = cur.fetchall()
    finally:
        conn.close()

    if not messages:
        return []

    conversations = []
    current_conversation = [messages[0]]
    for i in range(1, len(messages)):
        if messages[i][3] - messages[i-1][3] > timedelta(minutes=5):
            conversations.append(current_conversation)
            current_conversation = []
        current_conversation.append(messages[i])
    conversations.append(current_conversation)

    result_conversations = []
    for conv in conversations:
        result_conversations.append([{'role': msg[1], 'content': msg[2], 'timestamp': msg[3]} for msg in conv])

    result_conversations.sort(key=lambda conv: conv[0]['timestamp'], reverse=True)
    return result_conversations

# Main Streamlit app
def main():
    st.title("Jarvis - An AI Chatbot")
    st.write("Ask me anything about Marvel characters from the MCU or upload PDF/CSV files for additional data!")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "listening" not in st.session_state:
        st.session_state.listening = False
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None
    if "image_result" not in st.session_state:
        st.session_state.image_result = None
    if "timeline_data" not in st.session_state:
        st.session_state.timeline_data = None
    if "quiz_result" not in st.session_state:
        st.session_state.quiz_result = None
    if "variant_result" not in st.session_state:
        st.session_state.variant_result = None
    if "selected_history_id" not in st.session_state:
        st.session_state.selected_history_id = None
    if 'history_conversations' not in st.session_state:
        st.session_state.history_conversations = None

    # File upload (PDF and CSV)
    uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)
    
    # Initial data load
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = True

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # Voice input controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Voice Input", key="start_voice"):
            st.session_state.listening = True
            with st.spinner("Listening..."):
                prompt = take_command()
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)
                    store_chat_history("user", prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            phase_three_docs = retrieve_phase_three_documents(prompt)
                            pdf_docs = retrieve_file_documents(prompt, "marvel_characters_phase_three_documents")
                            all_docs = phase_three_docs + pdf_docs
                            if not all_docs or all(not doc.strip() for doc in all_docs):
                                response = "I'm sorry, I can only answer questions related to the Marvel characters in my database. Please ask about a Marvel character."
                            else:
                                response = generate_response(all_docs, st.session_state.messages)
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            store_chat_history("assistant", response)
                            talk(response)
                    st.session_state.listening = False
    with col2:
        if st.button("Stop Listening", key="stop_voice"):
            if st.session_state.get("listening", False):
                st.session_state.listening = False
                st.success("Stopped listening.")
            else:
                st.warning("Not currently listening.")
    with col3:
        if st.button("New Chat", key="new_chat"):
            st.session_state.messages = []
            st.rerun()

    # Text input field for user query
    if prompt := st.chat_input("What would you like to know about a Marvel character?", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        store_chat_history("user", prompt)

        with st.chat_message("assistant"):
            phase_three_docs = retrieve_phase_three_documents(prompt)
            pdf_docs = retrieve_file_documents(prompt, "marvel_characters_phase_three_documents")
            all_docs = phase_three_docs + pdf_docs
            if not all_docs or all(not doc.strip() for doc in all_docs):
                response = "I'm sorry, I can only answer questions related to the Marvel characters in my database. Please ask about a Marvel character."
            else:
                response = generate_response(all_docs, st.session_state.messages)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            store_chat_history("assistant", response)
            # No talk() for typed input

    # Additional Features
    st.sidebar.title("Additional Features")
    st.sidebar.subheader("View Chat History")
    if st.sidebar.button("Show Chat History", key="show_history"):
        st.session_state.history_conversations = retrieve_chat_conversations()

    if st.session_state.history_conversations is not None:
        conversations = st.session_state.history_conversations
        if conversations:
            options = {f"{conv[0]['timestamp'].strftime('%Y-%m-%d %H:%M')} - {conv[0]['content'][:30]}...": conv for conv in conversations}
            
            selected_key = st.sidebar.selectbox(
                "Select a conversation to load:",
                [""] + list(options.keys()),
                key="history_selectbox"
            )

            if selected_key:
                selected_conversation = options[selected_key]
                st.session_state.messages = [{'role': msg['role'], 'content': msg['content']} for msg in selected_conversation]
                st.session_state.history_conversations = None # Hide history after loading
                st.rerun()
        else:
            st.sidebar.write("No chat history available.")
            st.session_state.history_conversations = None # Reset

    st.sidebar.subheader("Compare Characters")

    # Character Comparison Tool
    with st.sidebar.form(key="compare_form"):
        characters = get_character_names()
        char1 = st.selectbox("Select First Character", characters, key="char1_select")
        char2 = st.selectbox("Select Second Character", characters, key="char2_select")
        submit_button = st.form_submit_button("Compare")
        if submit_button:
            print(f"Comparing {char1} and {char2}")  # Debug
            comparison = compare_characters(char1, char2)
            st.session_state.comparison_result = comparison
    if st.session_state.comparison_result:
        st.sidebar.write(st.session_state.comparison_result)
    st.sidebar.subheader("Character Images")
 
     # Image Display
    with st.sidebar.form(key="image_form"):
        character = st.selectbox("Select Character", get_character_names(), key="image_char_select")
        submit_button = st.form_submit_button("Display Image")
        if submit_button:
            print(f"Displaying image for {character}")  # Debug
            conn = get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT character_name, actor
                        FROM marvel_characters_phase_three
                        WHERE character_name = %s
                        LIMIT 1
                    """, (character,))
                    result = cur.fetchone()
                    print(f"Image query result for {character}: {result}")  # Debug
                if result:
                    name, actor = result
                    st.session_state.image_result = (name, actor)
            finally:
                conn.close()
    if st.session_state.image_result:
        name, actor = st.session_state.image_result
        st.sidebar.image("https://via.placeholder.com/150", caption=f"{name} (Actor: {actor or 'N/A'})")
    st.sidebar.subheader("Character Debuts Timeline")
    
    # Timeline View
    with st.sidebar.form(key="timeline_form"):
        submit_button = st.form_submit_button("Show Timeline")
        if submit_button:
            print("Generating timeline")  # Debug
            conn = get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT character_name, debut_year FROM marvel_characters_phase_three WHERE debut_year IS NOT NULL")
                    data = cur.fetchall()
                    print(f"Timeline data: {data}")  # Debug
                if data:
                    df = pd.DataFrame(data, columns=['character_name', 'debut_year'])
                    df = df.set_index('character_name')
                    st.session_state.timeline_data = df
                else:
                    st.session_state.timeline_data = None
            finally:
                conn.close()
    if st.session_state.timeline_data is not None:
        st.sidebar.bar_chart(st.session_state.timeline_data)
    st.sidebar.subheader("Marvel Quiz")
    
    # Quiz Mode
    with st.sidebar.form(key="quiz_form"):
        # Generate a new question if not present or after submission
        if "quiz_question" not in st.session_state or st.session_state.quiz_question is None:
            st.session_state.quiz_question, st.session_state.correct_answer = generate_quiz_question()
        question, correct_answer = st.session_state.quiz_question, st.session_state.correct_answer
        if isinstance(question, str):  # Only render input if question is a string
            user_answer = st.text_input("Question: " + question, key="quiz_answer_input")
            submit_button = st.form_submit_button("Submit Answer")
            if submit_button:
                print(f"User answer: {user_answer}, Correct answer: {correct_answer}")  # Debug
                if user_answer.lower() == correct_answer.lower():
                    st.session_state.quiz_result = "Correct!"
                else:
                    st.session_state.quiz_result = f"Wrong! The correct answer is {correct_answer}"
                st.session_state.quiz_question = None  # Reset for next question
        else:
            st.write("No quiz question available at the moment.")
    if st.session_state.quiz_result:
        st.sidebar.write(st.session_state.quiz_result)
    st.sidebar.subheader("Multiverse Variants")
    
    # Multiverse Variants
    with st.sidebar.form(key="variant_form"):
        character = st.selectbox("Select Character", get_character_names(), key="variant_char_select")
        submit_button = st.form_submit_button("Generate Variant")
        if submit_button:
            print(f"Generating variant for {character}")  # Debug
            prompt = f"Create a creative multiverse variant for {character} with unique powers and backstory."
            try:
                response = client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                ).choices[0].message.content.strip()
                st.session_state.variant_result = response
            except Exception as e:
                st.session_state.variant_result = f"Error generating variant: {e}"
    if st.session_state.variant_result:
        st.sidebar.write(st.session_state.variant_result)
    st.sidebar.subheader("Add to Favourites")
    
    # Save Favorites
    with st.sidebar.form(key="favorite_form"):
        character = st.selectbox("Select Character to Save", get_character_names(), key="favorite_char_select")
        submit_button = st.form_submit_button("Add to Favorites")
        if submit_button:
            print(f"Adding {character} to favorites")  # Debug
            if character not in st.session_state.favorites:
                st.session_state.favorites.append(character)
                st.sidebar.success(f"Added {character} to favorites.")
    st.sidebar.write("Favorites:", ", ".join(st.session_state.favorites))

    # Real-Time Updates
    if st.sidebar.button("Real-Time Updates", key="real_time_updates"):
        if time.time() - st.session_state.last_update > 300:  # Update every 5 minutes
            print("Performing real-time update")  # Debug
            conn = get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT character_name, real_name, actor, powers, debut_movie, debut_year, team_affiliation, status FROM marvel_characters_phase_three")
                    rows = cur.fetchall()
                    for row in rows:
                        content = f"Name: {row[0]}, Real Name: {row[1]}, Actor: {row[2]}, Powers: {row[3]}, Debut Movie: {row[4]}, Debut Year: {row[5]}, Team: {row[6]}, Status: {row[7]}"
                        embedding = get_embedding(content)
                        cur.execute("""
                            UPDATE marvel_characters_phase_three
                            SET embedding = %s::vector
                            WHERE character_name = %s
                        """, (embedding.tolist(), row[0]))
                    conn.commit()
                st.session_state.last_update = time.time()
                st.sidebar.success("Embeddings updated.")
            except Exception as e:
                st.sidebar.error(f"Error updating embeddings: {e}")
                conn.rollback()
            finally:
                conn.close()
        else:
            st.sidebar.write(f"Last update: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update))}. Next update in {int(300 - (time.time() - st.session_state.last_update))} seconds.")

if __name__ == "__main__":
    main()
