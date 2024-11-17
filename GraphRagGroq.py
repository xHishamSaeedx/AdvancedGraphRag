# Core langchain imports for building processing pipelines and chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# LangChain Groq integration
from langchain_groq import ChatGroq

# Document loading and processing utilities
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Database tools
from neo4j import GraphDatabase

# Python standard library imports
import os
from typing import List

from config import (
    GROQ_API_KEY,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    GROQ_MODEL
)

# Add utils imports
from utils import (
    extract_entities,
    create_entity_relationships,
    process_document_batch,
    save_retrieval_results
)

PDF_FILE = "txt_file_2.pdf"

# Set environment variables
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

def initialize_and_load_pdf():
    """Load PDF content into knowledge graph"""
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    # Initialize Groq
    llm = ChatGroq(
        temperature=0,
        model_name=GROQ_MODEL
    )
    
    # Load and process PDF
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create knowledge graph
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Process documents in batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            process_document_batch(
                batch_chunks=batch_chunks,
                session=session,
                llm=llm,
                start_idx=i
            )
            print(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
    
    driver.close()
    print(f"Loaded {len(chunks)} chunks into knowledge graph")
    return chunks

def ask_question(question: str):
    """Ask a question using knowledge graph"""
    llm = ChatGroq(
        temperature=0,
        model_name=GROQ_MODEL
    )
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        # Extract entities from question
        question_entities = extract_entities(question, llm)
        print(f"Extracted entities: {question_entities}")  # Debug print
        
        # Enhanced graph query to get more context
        graph_query = """
        MATCH (d:Document)
        WITH d
        MATCH (d)-[:CONTAINS]->(e:Entity)
        WHERE any(term IN $entities WHERE toLower(e.name) CONTAINS toLower(term))
           OR any(term IN $entities WHERE toLower(term) CONTAINS toLower(e.name))
        WITH d, e
        MATCH (d)-[:CONTAINS]->(related:Entity)
        WHERE related <> e
        WITH d, e, collect(DISTINCT related.name) as related_entities
        OPTIONAL MATCH (e)-[:CO_OCCURS_WITH]-(co:Entity)
        WITH d, e, related_entities, collect(DISTINCT co.name) as co_occurring
        RETURN 
            d.content as context,
            e.name as main_entity,
            related_entities,
            co_occurring
        ORDER BY size(related_entities) + size(co_occurring) DESC
        LIMIT 5
        """
        
        print(f"Searching for entities: {question_entities}")
        graph_result = session.run(graph_query, {"entities": question_entities})
        
        # Process and format graph results
        graph_contexts = []
        for record in graph_result:
            context = record["context"]
            main_entity = record["main_entity"]
            related = record["related_entities"]
            co_occurring = record["co_occurring"]
            
            graph_contexts.append({
                "content": context,
                "entity": main_entity,
                "related": related,
                "co_occurring": co_occurring
            })
        
        print(f"Found {len(graph_contexts)} relevant contexts")  # Debug print
        
        # Format context for the prompt
        formatted_contexts = []
        for ctx in graph_contexts:
            formatted_contexts.append(
                f"Information about {ctx['entity']}:\n"
                f"{ctx['content']}\n"
                f"Related concepts: {', '.join(ctx['related'])}\n"
                f"Co-occurring concepts: {', '.join(ctx['co_occurring'])}"
            )
        
        context = "\n\n".join(formatted_contexts) if formatted_contexts else "No relevant information found."
    
    # Save retrieval results for analysis
    save_retrieval_results(
        question=question,
        vector_contexts=[],  # Empty as we're not using vector store
        graph_contexts=graph_contexts,
        merged_context=context,
        llm_type="groq"
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a PDF document about college admissions and related topics.
Use the provided knowledge graph context to answer the question.

Context from Knowledge Graph:
{context}

Question: {question}

Instructions:
1. Use the provided context and relationships to give comprehensive answers
2. If the information isn't available, say "The document doesn't contain information about [topic]"
3. Be specific and cite relevant details from the document
4. Keep answers clear and concise

Answer:""")
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "context": context,
        "question": question
    })

if __name__ == "__main__":
    # Initialize and load PDF
    print(f"Loading PDF file: {PDF_FILE}...")
    chunks = initialize_and_load_pdf()
    print("PDF loaded successfully!")
    
    print("\nStarting Q&A session (type 'exit' to quit)")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'exit':
            break
            
        if not question:
            print("Please enter a valid question.")
            continue
            
        try:
            answer = ask_question(question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}") 