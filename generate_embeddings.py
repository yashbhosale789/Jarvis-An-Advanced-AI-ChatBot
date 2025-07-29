import os
import psycopg2
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
import json

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

# Generate embeddings for marvel_characters_phase_three
def generate_phase_three_embeddings():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT character_id, character_name, real_name, powers, actor FROM marvel_characters_phase_three WHERE embedding IS NULL")
            records = cur.fetchall()
            if records:
                for character_id, character_name, real_name, powers, actor in records:
                    text_to_embed = f"{character_name}, {real_name}, {powers}, {actor or ''}"
                    embedding = get_embedding(text_to_embed)
                    cur.execute(
                        "UPDATE marvel_characters_phase_three SET embedding = %s::vector WHERE character_id = %s",
                        (embedding.tolist(), character_id)
                    )
                conn.commit()
                print(f"Generated embeddings for {len(records)} phase three characters.")
    except Exception as e:
        print(f"Error generating phase three embeddings: {e}")
        conn.rollback()
    finally:
        conn.close()

# Generate embeddings for marvel_characters_phase_three_documents
def generate_csv_embeddings(file_path):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                content = json.dumps({
                    "character_id": row["character_id"],
                    "character_name": row["character_name"],
                    "real_name": row["real_name"],
                    "actor": row["actor"],
                    "powers": row["powers"],
                    "debut_movie": row["debut_movie"],
                    "debut_year": row["debut_year"],
                    "team_affiliation": row["team_affiliation"],
                    "status": row["status"]
                })
                embedding = get_embedding(content)
                cur.execute("""
                    INSERT INTO marvel_characters_phase_three_documents (file_name, character_id, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (file_name, character_id) DO UPDATE
                    SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """, (os.path.basename(file_path), row["character_id"], content, embedding.tolist()))
            conn.commit()
            print(f"Generated and stored embeddings for {len(df)} characters from CSV: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error generating CSV embeddings: {e}")
        conn.rollback()
    finally:
        conn.close()

# Generate embeddings for marvel_characters_phase_three_documents
def generate_pdf_embeddings(file_path):
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
                (os.path.basename(file_path), content, embedding.tolist())
            )
            conn.commit()
            print(f"Generated and stored embeddings for PDF: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error generating PDF embeddings: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    # Generate embeddings for phase three (only for null embeddings)
    generate_phase_three_embeddings()

    # Process the provided files
    csv_file = "marvel_characters.csv"
    pdf_file = "Marvel_Character_Brochure.pdf"
    generate_csv_embeddings(csv_file)
    generate_pdf_embeddings(pdf_file)
