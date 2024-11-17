import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from difflib import SequenceMatcher
import re
import json
from datetime import datetime
from pathlib import Path

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

def extract_entities(text: str, llm) -> List[str]:
    """Extract entities from text using LLM."""
    entity_extraction_prompt = ChatPromptTemplate.from_template("""
    Extract key entities from the following text about college admissions.
    Focus on:
    - College names
    - Application processes
    - Required documents
    - Important deadlines
    - Test names (like EAMCET, SAT)
    - Academic terms
    
    Return ONLY the entities as a simple list, one per line.
    Do not include explanations or categories.
    
    Text: {text}
    
    Entities:""")
    
    chain = entity_extraction_prompt | llm | StrOutputParser()
    result = chain.invoke({"text": text})
    entities = [entity.strip() for entity in result.split('\n') if entity.strip()]
    print(f"Extracted entities: {entities}")  # Debug print
    return entities

def create_entity_relationships(text: str, entities: List[str], session) -> None:
    """Create relationships between entities that co-occur in the same text."""
    if not entities:  # Skip if no entities found
        return
        
    try:
        # Create Document node with content
        doc_query = """
        CREATE (d:Document {content: $content})
        RETURN id(d) as doc_id
        """
        doc_result = session.run(doc_query, {"content": text})
        doc_id = doc_result.single()["doc_id"]
        print(f"Created document node with ID: {doc_id}")  # Debug print
        
        # Create Entity nodes and CONTAINS relationships
        for entity in entities:
            entity_query = """
            MERGE (e:Entity {name: $name})
            WITH e
            MATCH (d:Document) WHERE id(d) = $doc_id
            MERGE (d)-[:CONTAINS]->(e)
            RETURN e.name as entity_name
            """
            result = session.run(entity_query, {
                "name": entity,
                "doc_id": doc_id
            })
            print(f"Created/Merged entity: {result.single()['entity_name']}")  # Debug print
        
        # Create CO_OCCURS_WITH relationships between entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relationship_query = """
                MATCH (e1:Entity {name: $entity1})
                MATCH (e2:Entity {name: $entity2})
                MERGE (e1)-[r:CO_OCCURS_WITH]-(e2)
                RETURN type(r) as rel_type
                """
                result = session.run(relationship_query, {
                    "entity1": entity1,
                    "entity2": entity2
                })
                print(f"Created relationship between {entity1} and {entity2}")  # Debug print
                
    except Exception as e:
        print(f"Error in create_entity_relationships: {str(e)}")
        raise

def process_document_batch(batch_chunks, session, llm, start_idx):
    """Process a batch of document chunks for knowledge graph creation."""
    for i, chunk in enumerate(batch_chunks):
        print(f"\nProcessing chunk {start_idx + i}")  # Debug print
        try:
            # Extract entities
            entities = extract_entities(chunk.page_content, llm)
            print(f"Found {len(entities)} entities")  # Debug print
            
            if entities:
                # Create relationships in Neo4j
                create_entity_relationships(chunk.page_content, entities, session)
                print(f"Created relationships for chunk {start_idx + i}")  # Debug print
            else:
                print(f"No entities found in chunk {start_idx + i}")  # Debug print
                
        except Exception as e:
            print(f"Error processing chunk {start_idx + i}: {str(e)}")
            continue  # Continue with next chunk even if one fails

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts"""
    return SequenceMatcher(None, text1, text2).ratio()

def clean_text(text: str) -> str:
    """Clean text for comparison"""
    return re.sub(r'\s+', ' ', text.lower().strip())

def merge_contexts(
    vector_contexts: List[str],
    graph_contexts: List[Dict[str, Any]],
    question: str,
    similarity_threshold: float = 0.8
) -> str:
    """
    Merge contexts from vector store and knowledge graph,
    removing redundancy and organizing by relevance
    """
    # Clean and deduplicate vector contexts
    cleaned_vector_contexts = [clean_text(ctx) for ctx in vector_contexts]
    unique_vector_contexts = []
    
    for i, ctx in enumerate(cleaned_vector_contexts):
        is_duplicate = False
        for existing in unique_vector_contexts:
            if calculate_similarity(ctx, existing) > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_vector_contexts.append(vector_contexts[i])
    
    # Clean and organize graph contexts
    graph_content = []
    entities_mentioned = set()
    
    for ctx in graph_contexts:
        content = clean_text(ctx["content"])
        # Check if content is too similar to vector contexts
        is_duplicate = any(
            calculate_similarity(content, vec_ctx) > similarity_threshold
            for vec_ctx in cleaned_vector_contexts
        )
        
        if not is_duplicate and ctx["entity"] not in entities_mentioned:
            entities_mentioned.add(ctx["entity"])
            graph_content.append(
                f"Information about {ctx['entity']}:\n{ctx['content']}\n"
                f"Related concepts: {', '.join(ctx['related'])}"
            )
    
    # Combine contexts with clear separation
    merged = []
    
    if unique_vector_contexts:
        merged.append("Relevant Document Sections:")
        merged.extend(unique_vector_contexts)
    
    if graph_content:
        if merged:
            merged.append("\nKnowledge Graph Connections:")
        merged.extend(graph_content)
    
    return "\n\n".join(merged)

def save_retrieval_results(
    question: str,
    vector_contexts: List[str],
    graph_contexts: List[Dict[str, Any]],
    merged_context: str,
    llm_type: str
) -> None:
    """Save retrieval results to files for analysis"""
    # Create logs directory if it doesn't exist
    log_dir = Path("retrieval_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Format the results
    results = {
        "question": question,
        "vector_store_results": vector_contexts,
        "knowledge_graph_results": [
            {
                "main_entity": ctx["entity"],
                "content": ctx["content"],
                "related_concepts": ctx["related"]
            }
            for ctx in graph_contexts
        ],
        "merged_context": merged_context,
        "timestamp": timestamp
    }
    
    # Save to file
    filename = log_dir / f"retrieval_results_{llm_type}_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nRetrieval results saved to: {filename}")