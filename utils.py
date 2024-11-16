import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

def create_optimized_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """Create an optimized text splitter for large documents"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

def create_optimized_faiss_index(chunks: List[Document], embeddings: Any, index_name: str, batch_size: int = 1000):
    """Create an optimized FAISS index for large datasets"""
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(chunks))
        batch_chunks = chunks[start_idx:end_idx]
        
        # Get embeddings for batch
        vectors = embeddings.embed_documents([chunk.page_content for chunk in batch_chunks])
        vectors = np.array(vectors, dtype=np.float32)
        
        if i == 0:
            # Initialize index with first batch
            dim = vectors.shape[1]
            n_clusters = min(int(np.sqrt(len(chunks))), 256)  # Cap maximum clusters
            
            # Create IVF index
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_L2)
            
            # Train the index
            if not index.is_trained:
                index.train(vectors)
            
            # Create vector store
            vector_store = FAISS.from_documents(batch_chunks, embeddings)
            vector_store.index = index
        else:
            # Load and update existing index
            vector_store = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
        
        # Add vectors to index
        vector_store.index.add(vectors)
        vector_store.save_local(index_name)
        
        print(f"Processed batch {i+1}/{total_batches}")
    
    return vector_store

def optimized_similarity_search(question: str, vector_store: FAISS, k: int = 3, nprobe: int = 10):
    """Perform optimized similarity search for large datasets"""
    # Set number of clusters to probe
    vector_store.index.nprobe = nprobe
    
    # Get similar documents
    similar_docs = vector_store.similarity_search(
        question,
        k=k,
        fetch_k=2*k  # Fetch more candidates for better results
    )
    
    return similar_docs

def get_entity_prompt(llm: Any, content: str):
    """Get appropriate entity extraction prompt based on LLM type"""
    if isinstance(llm, ChatGoogleGenerativeAI):
        # For Google's Gemini, use only human message
        return ChatPromptTemplate.from_messages([
            ("human", """You are an AI assistant that extracts key entities and their relationships from text.
                        Focus on: Colleges, Courses, Requirements, Processes, Documents
                        
                        Text to analyze:
                        """ + content)
        ])
    else:
        # For other LLMs (OpenAI, Groq), use system and human messages
        return ChatPromptTemplate.from_messages([
            ("system", """Extract key entities and their relationships from the text.
                         Focus on: Colleges, Courses, Requirements, Processes, Documents"""),
            ("human", content)
        ])

def process_document_batch(
    batch_chunks: List[Document],
    session: Any,
    llm: Any,
    start_idx: int
):
    """Process a batch of document chunks for the knowledge graph"""
    for i, chunk in enumerate(batch_chunks):
        chunk_idx = start_idx + i
        try:
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
                    "id": f"chunk_{chunk_idx}",
                    "content": chunk.page_content,
                    "page": chunk.metadata.get('page', 0)
                }
            )
            
            # Get appropriate prompt based on LLM type
            entity_prompt = get_entity_prompt(llm, chunk.page_content)
            
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
                    "chunk_id": f"chunk_{chunk_idx}",
                    "entities": entities.split('\n')
                }
            )
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {str(e)}")
            continue