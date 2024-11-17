# Core langchain imports for building processing pipelines and chains
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_core.pydantic_v1 import BaseModel, Field

# LangChain community imports for graph and vector operations
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# LangChain Google AI integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LangChain experimental features for graph operations
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Document loading and processing utilities
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

# Database and visualization tools
from neo4j import GraphDatabase

# Python standard library imports
import os
from typing import Tuple, List, Optional

from config import (
    GOOGLE_API_KEY,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    GOOGLE_MODEL
)

# Add imports
from langchain.vectorstores import FAISS

# Add utils imports
from utils import (
    create_optimized_text_splitter,
    create_optimized_faiss_index,
    optimized_similarity_search,
    process_document_batch,
    extract_entities,
    create_entity_relationships,
    merge_contexts
)

PDF_FILE = "txt_file_2.pdf"

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
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
    """Load PDF content and store with both optimized FAISS vector store and Neo4j knowledge graph"""
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    # Initialize Google embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Load and process PDF with optimized text splitter
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()
    
    text_splitter = create_optimized_text_splitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create optimized FAISS vector store
    vector_store = create_optimized_faiss_index(
        chunks=chunks,
        embeddings=embeddings,
        index_name="faiss_index_google",
        batch_size=1000
    )
    
    # Initialize Google's Generative AI for entity extraction
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        temperature=0
    )
    
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Process documents in batches
        batch_size = 50  # Adjust based on your needs
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
    print(f"Loaded {len(chunks)} chunks from PDF with optimized knowledge graph and vector store")
    return chunks

def ask_question(question: str):
    """Ask a question using both knowledge graph and vector store"""
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        temperature=0
    )
    
    # Get vector store results
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.load_local(
        "faiss_index_google", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    similar_docs = optimized_similarity_search(
        question=question,
        vector_store=vector_store,
        k=3
    )
    vector_contexts = [doc.page_content for doc in similar_docs]
    
    # Get knowledge graph results
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        question_entities = extract_entities(question, llm)
        
        graph_query = """
        MATCH (e:Entity)
        WHERE e.name IN $entities OR any(term IN $entities WHERE e.name CONTAINS term)
        WITH e
        MATCH (d:Document)-[:CONTAINS]->(e)
        WITH d, e
        MATCH (d)-[:CONTAINS]->(related:Entity)
        WHERE related <> e
        WITH d, e, collect(DISTINCT related.name) as related_entities
        RETURN d.content as context, e.name as main_entity, related_entities
        ORDER BY size(related_entities) DESC
        LIMIT 5
        """
        
        graph_result = session.run(graph_query, {"entities": question_entities})
        
        graph_contexts = []
        for record in graph_result:
            context = record["context"]
            main_entity = record["main_entity"]
            related = record["related_entities"]
            graph_contexts.append({
                "content": context,
                "entity": main_entity,
                "related": related
            })
    
    # Merge contexts from both sources
    merged_context = merge_contexts(
        vector_contexts=vector_contexts,
        graph_contexts=graph_contexts,
        question=question
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a PDF document about college admissions and related topics.
Use the combined context from vector search and knowledge graph to answer the question.

Context:
{merged_context}

Question: {question}

Instructions:
1. Use the provided context to give comprehensive answers
2. If the information isn't available, say "The document doesn't contain information about [topic]"
3. Be specific and cite relevant details from the document
4. Keep answers clear and concise

Answer:""")
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "merged_context": merged_context,
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