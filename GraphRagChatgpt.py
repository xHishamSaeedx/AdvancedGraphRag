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

# Add FAISS import
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
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
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
        index_name="faiss_index_chatgpt",
        batch_size=1000
    )
    
    # Initialize ChatGPT for entity extraction
    llm = ChatOpenAI(temperature=0, model_name=OPENAI_MODEL)
    
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
    llm = ChatOpenAI(temperature=0, model_name=OPENAI_MODEL)
    
    # Get vector store results
    print("\nQuerying vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "faiss_index_chatgpt", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    similar_docs = vector_store.similarity_search(
        question,
        k=3
    )
    vector_contexts = [doc.page_content for doc in similar_docs]
    print(f"Found {len(vector_contexts)} relevant documents from vector store")
    
    # Get knowledge graph results
    print("\nQuerying knowledge graph...")
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        # Extract entities from question
        question_entities = extract_entities(question, llm)
        print(f"Extracted entities: {question_entities}")
        
        # Query knowledge graph
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
        
        print(f"Found {len(graph_contexts)} relevant contexts from knowledge graph")
    
    # Merge contexts from both sources
    merged_context = merge_contexts(
        vector_contexts=vector_contexts,
        graph_contexts=graph_contexts,
        question=question
    )
    
    # Save retrieval results
    save_retrieval_results(
        question=question,
        vector_contexts=vector_contexts,
        graph_contexts=graph_contexts,
        merged_context=merged_context,
        llm_type="chatgpt"
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