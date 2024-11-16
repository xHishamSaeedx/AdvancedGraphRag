# Core langchain imports for building processing pipelines and chains
from langchain_core.runnables import (
    RunnableBranch,      # For conditional branching in chains
    RunnableLambda,      # For parallel execution of chain components
    RunnableParallel,    # For parallel execution of chain components
    RunnablePassthrough, # For passing inputs through without modification
)
from langchain_core.prompts import ChatPromptTemplate    # For creating chat-based prompts
from langchain_core.prompts.prompt import PromptTemplate # For creating text-based prompts
from langchain_core.messages import AIMessage, HumanMessage  # For formatting chat messages
from langchain_core.output_parsers import StrOutputParser    # For parsing string outputs
from langchain_core.runnables import ConfigurableField       # For configurable chain components
from langchain_core.pydantic_v1 import BaseModel, Field      # For data validation and settings

# LangChain community imports for graph and vector operations
from langchain_community.graphs import Neo4jGraph            # For Neo4j graph database operations
from langchain_community.vectorstores import Neo4jVector     # For vector storage in Neo4j
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars  # For text cleaning

# LangChain OpenAI integration
from langchain_openai import ChatOpenAI        # For interacting with OpenAI chat models
from langchain_openai import OpenAIEmbeddings  # For creating text embeddings using OpenAI

# LangChain experimental features for graph operations
from langchain_experimental.graph_transformers import LLMGraphTransformer  # For graph-based LLM operations

# Document loading and processing utilities
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter  # For splitting text into smaller chunks

# Database and visualization tools
from neo4j import GraphDatabase           # For direct Neo4j database connections

# Python standard library imports
import os                          # For operating system operations and env variables
from typing import Tuple, List, Optional  # For type hints and annotations

from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_MODEL
PDF_FILE = "txt_file_2.pdf"  # Hardcoded PDF file name

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
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
    """Load PDF content and store in Neo4j with knowledge graph"""
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    # Load PDF document
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Initialize LLM for entity extraction
    llm = ChatOpenAI(temperature=0, model_name=OPENAI_MODEL)
    
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Store each chunk as a document node
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
                ("system", """Extract key entities and their relationships from the text.
                             Focus on: Colleges, Courses, Requirements, Processes, Documents"""),
                ("human", chunk.page_content)
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
    print(f"Loaded {len(chunks)} chunks from PDF with knowledge graph")
    return chunks

def ask_question(question: str):
    """Ask a question using both document content and knowledge graph"""
    llm = ChatOpenAI(temperature=0, model_name=OPENAI_MODEL)
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        # Get document content
        doc_result = session.run(
            "MATCH (d:Document) RETURN d.content AS content ORDER BY d.page"
        )
        doc_context = "\n".join([record["content"] for record in doc_result])
        
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
Use the information provided in both the document content and knowledge graph context to answer the question.

Document Content:
{doc_context}

Knowledge Graph Context:
{graph_context}

Question: {question}

Instructions:
1. Use both document content and knowledge graph relationships to provide comprehensive answers
2. If the information isn't available, say "The document doesn't contain information about [topic]"
3. Be specific and cite relevant details from the document
4. Keep answers clear and concise

Answer:""")
    
    # Get response
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "doc_context": doc_context,
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