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
    """Load PDF content and store with both FAISS vector store and Neo4j knowledge graph"""
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    # Initialize Google embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Load and process PDF
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()
    
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_google")
    
    # Initialize Google's Generative AI for entity extraction
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        temperature=0
    )
    
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Process each chunk for knowledge graph
        for i, chunk in enumerate(chunks):
            # Store document content
            session.run(
                """
                CREATE (d:Document {
                    id: $id,
                    content: $content,
                    page: $page
                })
                """,
                {
                    "id": f"chunk_{i}",
                    "content": chunk.page_content,
                    "page": chunk.metadata.get('page', 0)
                }
            )
            
            # Extract entities and relationships
            entity_prompt = ChatPromptTemplate.from_messages([
                ("human", """You are an AI assistant that extracts key entities and their relationships from text.
                            Focus on: Colleges, Courses, Requirements, Processes, Documents
                            
                            Text to analyze:
                            """ + chunk.page_content)
            ])
            
            entity_chain = entity_prompt | llm | StrOutputParser()
            entities = entity_chain.invoke({})
            
            # Create entity nodes and relationships
            session.run(
                """
                MATCH (d:Document {id: $chunk_id})
                WITH d
                UNWIND $entities as entity
                MERGE (e:Entity {name: entity})
                CREATE (d)-[:CONTAINS]->(e)
                """,
                {
                    "chunk_id": f"chunk_{i}",
                    "entities": entities.split('\n')
                }
            )
    
    driver.close()
    print(f"Loaded {len(chunks)} chunks from PDF with knowledge graph and vector store")
    return chunks

def ask_question(question: str):
    """Ask a question using vector similarity and knowledge graph"""
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        temperature=0
    )
    
    # Load vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.load_local(
        "faiss_index_google", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Get relevant documents using vector similarity
    similar_docs = vector_store.similarity_search(question, k=3)
    vector_context = "\n".join([doc.page_content for doc in similar_docs])
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        # Get relevant entities and their relationships
        graph_result = session.run(
            """
            MATCH (d:Document)-[:CONTAINS]->(e:Entity)
            WHERE e.name =~ $query
            WITH d, e
            MATCH (d)-[:CONTAINS]->(related:Entity)
            RETURN DISTINCT e.name as entity, collect(DISTINCT related.name) as related
            LIMIT 5
            """,
            {"query": f"(?i).*{question}.*"}
        )
        
        # Format graph context
        graph_context = []
        for record in graph_result:
            entity = record["entity"]
            related = record["related"]
            graph_context.append(f"Entity: {entity}\nRelated concepts: {', '.join(related)}")
        
        graph_context = "\n".join(graph_context)
    
    # Create combined prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a PDF document about college admissions and related topics.
Use the information provided in the vector similarity results and knowledge graph context to answer the question.

Vector Similar Content:
{vector_context}

Knowledge Graph Context:
{graph_context}

Question: {question}

Instructions:
1. Use both vector similarity results and knowledge graph relationships to provide comprehensive answers
2. If the information isn't available, say "The document doesn't contain information about [topic]"
3. Be specific and cite relevant details from the document
4. Keep answers clear and concise

Answer:""")
    
    # Get response
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "vector_context": vector_context,
        "graph_context": graph_context,
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