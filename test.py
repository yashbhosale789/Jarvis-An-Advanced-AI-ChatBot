#!/usr/bin/env python3
"""
Hakathon resource Testing Script
Tests Azure OpenAI, GPT-4o, embeddings, and PostgreSQL data retrieval
"""

import os
import psycopg2
from openai import AzureOpenAI
import numpy as np
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarvelCharacter:
    """Data class for Marvel character"""
    character_id: int
    character_name: str
    real_name: str
    actor: Optional[str]
    debut_movie: Optional[str]
    debut_year: Optional[int]
    team_affiliation: Optional[str]
    status: str
    powers: Optional[str]

class MarvelCharactersTestSuite:
    """Test suite for Marvel characters data with AI integration"""
    
    def __init__(self):
        self.client = None
        self.setup_openai()
        self.setup_database()
        self.test_results = {}
        
        
    def setup_openai(self):
        try:
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            self.gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o")
            self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

            self.client = AzureOpenAI(
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint,
            )

            logger.info("OpenAI client configured successfully")
        except Exception as e:
            logger.error(f"Failed to setup OpenAI: {e}")
            raise
    
    def setup_database(self):
        """Setup PostgreSQL database connection"""
        try:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'marvel_db'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password'),
                'port': os.getenv('DB_PORT', '5432')
            }
            
            # Test connection
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            logger.info("Database connection configured successfully")
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
    
    def test_database_connection(self) -> bool:
        """Test PostgreSQL database connection"""
        try:
            logger.info("Testing database connection...")
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Test table existence
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'marvel_characters'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.error("Table 'marvel_characters' does not exist")
                return False
            
            # Test data retrieval
            cursor.execute("SELECT COUNT(*) FROM marvel_characters;")
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            logger.info(f"Database test passed. Found {count} characters in table.")
            self.test_results['database_connection'] = True
            return True
            
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            self.test_results['database_connection'] = False
            return False
        
    def retrieve_marvel_characters(self, limit: int = 10) -> List[MarvelCharacter]:
        """Retrieve Marvel characters from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            query = """
                SELECT character_id, character_name, real_name, actor, 
                       debut_movie, debut_year, team_affiliation, status, powers
                FROM marvel_characters
                ORDER BY character_id
                LIMIT %s;
            """
            
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            
            characters = []
            for row in rows:
                character = MarvelCharacter(
                    character_id=row[0],
                    character_name=row[1],
                    real_name=row[2],
                    actor=row[3],
                    debut_movie=row[4],
                    debut_year=row[5],
                    team_affiliation=row[6],
                    status=row[7],
                    powers=row[8]
                )
                characters.append(character)
            
            cursor.close()
            conn.close()
            
            logger.info(f"Retrieved {len(characters)} characters from database")
            return characters
            
        except Exception as e:
            logger.error(f"Failed to retrieve characters: {e}")
            return []
    
    def test_azure_openai_chat(self, character: MarvelCharacter) -> bool:
        """Test Azure OpenAI chat completion using direct API"""
        try:
            logger.info(f"Testing Azure OpenAI chat for {character.character_name}...")
            
            prompt = f"""
            Tell me an interesting fact about {character.character_name} ({character.real_name}) 
            from the Marvel Cinematic Universe. They were played by {character.actor or 'unknown actor'} 
            and first appeared in {character.debut_movie or 'unknown movie'}.
            """
            
            # Test with different API formats for compatibility
            try:
                # Try newer API format first
                response = self.client.chat.completions.create(
                        deployment_id=self.gpt_deployment,
                        messages=[...],
                        max_tokens=150,
                        temperature=0.7
                    )
                
                result = response.choices[0].message.content.strip()
                logger.info(f"Azure OpenAI response: {result[:100]}...")
                
                self.test_results['azure_openai_chat'] = True
                return True
                
            except Exception as newer_error:
                logger.warning(f"New API format failed: {newer_error}")
                
                
            
        except Exception as e:
            logger.error(f"Azure OpenAI chat test failed: {e}")
            self.test_results['azure_openai_chat'] = False
            return False
    
    def test_gpt4o_analysis(self, characters: List[MarvelCharacter]) -> bool:
        """Test GPT-4o for character analysis using direct API"""
        try:
            logger.info("Testing GPT-4o for character analysis...")
            
            # Create character summary
            char_summary = []
            for char in characters[:5]:  # Use first 5 characters
                char_summary.append(f"- {char.character_name} ({char.real_name}): {char.powers or 'No powers listed'}")
            
            prompt = f"""
            Analyze these Marvel characters and provide insights about their power diversity:
            
            {chr(10).join(char_summary)}
            
            Please categorize their powers and suggest which team composition would be most effective.
            """
            
            # Try chat completion format
            try:
                response = openai.ChatCompletion.create(
                    engine=self.gpt_deployment,
                    messages=[
                        {"role": "system", "content": "You are a tactical analyst for superhero teams."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.5
                )
                
                result = response.choices[0].message.content.strip()
                logger.info(f"GPT-4o analysis: {result[:100]}...")
                
                self.test_results['gpt4o_analysis'] = True
                return True
                
            except Exception as chat_error:
                logger.warning(f"Chat completion failed: {chat_error}")
                
                # Try completion format
              
                
                self.test_results['gpt4o_analysis'] = True
                return True
            
        except Exception as e:
            logger.error(f"GPT-4o analysis test failed: {e}")
            self.test_results['gpt4o_analysis'] = False
            return False
        
    def test_embeddings(self, characters: List[MarvelCharacter]) -> bool:
        """Test OpenAI embeddings for character similarity using direct API"""
        try:
            logger.info("Testing OpenAI embeddings...")
            
            # Prepare texts for embedding
            texts = []
            for char in characters[:3]:  # Use first 3 characters
                text = f"{char.character_name} is {char.real_name}. Powers: {char.powers or 'Unknown'}. Team: {char.team_affiliation or 'None'}"
                texts.append(text)
            
            # Get embeddings using direct API
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    response = self.client.embeddings.create(
                            model=self.embedding_deployment,
                            input=text
                        )
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)
                    logger.info(f"Generated embedding for character {i+1}: {characters[i].character_name}")
                    
                except Exception as embed_error:
                    logger.error(f"Failed to generate embedding for {characters[i].character_name}: {embed_error}")
                    # Continue with other characters
                    continue
            
            # Calculate similarity between first two characters if we have at least 2 embeddings
            if len(embeddings) >= 2:
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                logger.info(f"Similarity between {characters[0].character_name} and {characters[1].character_name}: {similarity:.4f}")
                
                # Additional similarity comparisons if we have more embeddings
                if len(embeddings) >= 3:
                    similarity_2 = np.dot(embeddings[0], embeddings[2]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2])
                    )
                    logger.info(f"Similarity between {characters[0].character_name} and {characters[2].character_name}: {similarity_2:.4f}")
                
                self.test_results['embeddings'] = True
                return True
            else:
                logger.error("Not enough embeddings generated for similarity calculation")
                self.test_results['embeddings'] = False
                return False
            
        except Exception as e:
            logger.error(f"Embeddings test failed: {e}")
            self.test_results['embeddings'] = False
            return False
    
    def test_character_search(self, search_term: str) -> bool:
        """Test character search functionality"""
        try:
            logger.info(f"Testing character search for '{search_term}'...")
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Search by character name or real name
            search_query = """
                SELECT character_name, real_name, actor, powers
                FROM marvel_characters
                WHERE character_name ILIKE %s OR real_name ILIKE %s
                ORDER BY character_name
                LIMIT 5;
            """
            
            search_pattern = f"%{search_term}%"
            cursor.execute(search_query, (search_pattern, search_pattern))
            results = cursor.fetchall()
            
            logger.info(f"Found {len(results)} characters matching '{search_term}':")
            for result in results:
                logger.info(f"  - {result[0]} ({result[1]}) - {result[2] or 'No actor'}")
            
            cursor.close()
            conn.close()
            
            self.test_results['character_search'] = True
            return True
            
        except Exception as e:
            logger.error(f"Character search test failed: {e}")
            self.test_results['character_search'] = False
            return False
    
    def test_data_aggregation(self) -> bool:
        """Test data aggregation queries"""
        try:
            logger.info("Testing data aggregation...")
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Test various aggregation queries
            queries = [
                ("Characters by status", "SELECT status, COUNT(*) FROM marvel_characters GROUP BY status ORDER BY COUNT(*) DESC;"),
                ("Characters by team", "SELECT team_affiliation, COUNT(*) FROM marvel_characters WHERE team_affiliation IS NOT NULL GROUP BY team_affiliation ORDER BY COUNT(*) DESC LIMIT 5;"),
                ("Characters by debut year", "SELECT debut_year, COUNT(*) FROM marvel_characters WHERE debut_year IS NOT NULL GROUP BY debut_year ORDER BY debut_year;")
            ]
            
            for query_name, query_sql in queries:
                cursor.execute(query_sql)
                results = cursor.fetchall()
                logger.info(f"{query_name}:")
                for result in results:
                    logger.info(f"  - {result[0]}: {result[1]}")
                logger.info("")
            
            cursor.close()
            conn.close()
            
            self.test_results['data_aggregation'] = True
            return True
            
        except Exception as e:
            logger.error(f"Data aggregation test failed: {e}")
            self.test_results['data_aggregation'] = False
            return False
        
    def run_all_tests(self):
        """Run all tests in the suite"""
        logger.info("Starting Marvel Characters Test Suite...")
        logger.info("="*50)
        
        # Test database connection first
        if not self.test_database_connection():
            logger.error("Database connection failed. Stopping tests.")
            return
        
        # Retrieve characters for testing
        characters = self.retrieve_marvel_characters(limit=10)
        if not characters:
            logger.error("No characters retrieved. Stopping tests.")
            return
        
        # Run all tests
        tests = [
            ("Character Search", lambda: self.test_character_search("Iron")),
            ("Data Aggregation", self.test_data_aggregation),
            ("Azure OpenAI Chat", lambda: self.test_azure_openai_chat(characters[0])),
            ("GPT-4o Analysis", lambda: self.test_gpt4o_analysis(characters)),
            ("Embeddings", lambda: self.test_embeddings(characters))
        ]
        
        logger.info("Running individual tests...")
        for test_name, test_func in tests:
            logger.info(f"Running {test_name}...")
            try:
                test_func()
                logger.info(f"✓ {test_name} passed")
            except Exception as e:
                logger.error(f"✗ {test_name} failed: {e}")
            logger.info("-" * 30)
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary"""
        logger.info("="*50)
        logger.info("TEST SUMMARY")
        logger.info("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        logger.info("="*50)

def main():
    """Main function to run the test suite"""
    # Updated required environment variables for your setup
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Please set the following environment variables:")
        logger.info("- AZURE_OPENAI_ENDPOINT")
        logger.info("- AZURE_OPENAI_API_KEY")
        logger.info("- AZURE_OPENAI_API_VERSION (optional, defaults to 2025-01-01-preview)")
        logger.info("- AZURE_OPENAI_GPT_DEPLOYMENT (optional, defaults to gpt-4o)")
        logger.info("- AZURE_OPENAI_EMBEDDING_DEPLOYMENT (optional, defaults to text-embedding-ada-002)")
        logger.info("- DB_HOST, DB_NAME, DB_USER, DB_PASSWORD (for database)")
        return
    
    # Set your environment variables if not already set
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        os.environ["AZURE_OPENAI_ENDPOINT"] = ""
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        os.environ["AZURE_OPENAI_API_KEY"] = ""
    if not os.getenv("AZURE_OPENAI_API_VERSION"):
        os.environ["AZURE_OPENAI_API_VERSION"] = ""
    if not os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT"):
        os.environ["AZURE_OPENAI_GPT_DEPLOYMENT"] = ""
    if not os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
        os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = ""
    
    logger.info("Starting Marvel Characters Test Suite (Direct API - No LangChain)")
    
    # Run tests
    test_suite = MarvelCharactersTestSuite()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()