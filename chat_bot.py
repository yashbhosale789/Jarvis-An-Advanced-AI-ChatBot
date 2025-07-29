import os
import psycopg2
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import pandas as pd
import pdfplumber

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")  # 2025-01-01-preview
)

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
def get_embedding(text, model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")):  # text-embedding-ada-002
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)

# Retrieve data from marvel_characters_phase_three
def retrieve_phase_three_documents(query, top_k=3):
    query_embedding = get_embedding(query)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT character_name, real_name, powers, debut_movie, debut_year, team_affiliation, status, actor, embedding
                FROM marvel_characters_phase_three
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """, (query_embedding.tolist(), top_k))
            results = cur.fetchall()
        return [
            f"Character: {row[0]}, Real Name: {row[1]}, Powers: {row[2]}, Debut: {row[3]} ({row[4]}), Team: {row[5]}, Status: {row[6]}, Actor: {row[7] or 'N/A'}"
            for row in results
        ]
    finally:
        conn.close()

# Retrieve data from csv/pdf tables
def retrieve_file_documents(query, table_name, top_k=3):
    query_embedding = get_embedding(query)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT content, embedding FROM {table_name} ORDER BY embedding <-> %s::vector LIMIT %s",
                (query_embedding.tolist(), top_k)
            )
            results = cur.fetchall()
        return [row[0] for row in results]
    finally:
        conn.close()

# Process and store CSV file
def process_csv_file(file_path, file_name):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            df = pd.read_csv(file_path)
            # Insert or update phase three data
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO marvel_characters_phase_three (character_id, character_name, real_name, actor, powers, debut_movie, debut_year, team_affiliation, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (character_id) DO UPDATE SET character_name = EXCLUDED.character_name, real_name = EXCLUDED.real_name,
                    actor = EXCLUDED.actor, powers = EXCLUDED.powers, debut_movie = EXCLUDED.debut_movie,
                    debut_year = EXCLUDED.debut_year, team_affiliation = EXCLUDED.team_affiliation, status = EXCLUDED.status
                """, (row["character_id"], row["character_name"], row["real_name"], row["actor"], row["powers"],
                      row["debut_movie"], row["debut_year"], row["team_affiliation"], row["status"]))
            # Store CSV content and embedding
            content = df.to_csv(index=False)
            embedding = get_embedding(content)
            cur.execute(
                "INSERT INTO marvel_characters_phase_three_documents (file_name, content, embedding) VALUES (%s, %s, %s::vector) ON CONFLICT (file_name) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding",
                (file_name, content, embedding.tolist())
            )
            conn.commit()
            print(f"Processed and stored CSV: {file_name}")
    except Exception as e:
        print(f"Error processing CSV: {e}")
        conn.rollback()
    finally:
        conn.close()

# Process and store PDF file
def process_pdf_file(file_path, file_name):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            with pdfplumber.open(file_path) as pdf:
                content = "".join(page.extract_text() or "" for page in pdf.pages)
            if not content:
                content = "No extractable text found in PDF."
            embedding = get_embedding(content)
            cur.execute(
                "INSERT INTO marvel_characters_phase_three_documents (file_name, content, embedding) VALUES (%s, %s, %s::vector) ON CONFLICT (file_name) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding",
                (file_name, content, embedding.tolist())
            )
            conn.commit()
            print(f"Processed and stored PDF: {file_name}")
    except Exception as e:
        print(f"Error processing PDF: {e}")
        conn.rollback()
    finally:
        conn.close()

# Generate response
def generate_response(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"""You are a Marvel expert chatbot. Use the following Marvel character information to answer the user's query.
Character Information:
{context}

Query: {query}

Answer:"""
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT"),  # gpt-4o
        messages=[
            {"role": "system", "content": "You are a knowledgeable Marvel expert providing accurate and concise answers about Marvel characters. Only respond based on the provided character information and do not answer questions unrelated to the characters in the database."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Main chatbot function
def chatbot(query):
    phase_three_docs = retrieve_phase_three_documents(query)
    csv_docs = retrieve_file_documents(query, "marvel_characters_phase_three_documents")
    pdf_docs = retrieve_file_documents(query, "marvel_characters_phase_three_documents")
    all_docs = phase_three_docs + csv_docs + pdf_docs

    if not all_docs or all(not doc.strip() for doc in all_docs):
        return "I'm sorry, I can only answer questions related to the Marvel characters in my database. Please ask about a Marvel character."
    return generate_response(query, all_docs)

if __name__ == "__main__":
    print("Welcome to the Marvel Character Chatbot! Ask about your favorite Marvel characters.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = chatbot(user_input)
        print(f"Bot: {response}")
