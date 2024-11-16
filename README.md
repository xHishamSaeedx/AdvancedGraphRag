# Advanced Graph RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines knowledge graphs and vector embeddings, supporting multiple LLM providers (OpenAI, Google's Gemini, and Groq) for enhanced document question-answering.

## Features

- **Multiple LLM Support**:

  - OpenAI GPT (ChatGPT)
  - Google Gemini Pro
  - Groq (Mixtral-8x7b)

- **Hybrid Retrieval System**:

  - Knowledge Graph (Neo4j) for structured relationships
  - Vector Similarity (FAISS) for semantic search
  - Optimized batch processing for large documents

- **Advanced Optimizations**:

  - Efficient text chunking with RecursiveCharacterTextSplitter
  - FAISS IVF indexing for fast similarity search
  - Batched document processing
  - LLM-specific prompt handling

- **Multiple Interfaces**:
  - Command-line interface for each LLM
  - Streamlit web interface
  - Programmatic API

## Prerequisites

- Python 3.9+
- Neo4j Database
- API keys for desired LLM providers:
  - OpenAI API key
  - Google AI API key
  - Groq API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AdvancedGraphRag.git
cd AdvancedGraphRag
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4-turbo-preview

# Google Configuration
GOOGLE_API_KEY=your_google_key
GOOGLE_MODEL=gemini-pro

# Groq Configuration
GROQ_API_KEY=your_groq_key
GROQ_MODEL=mixtral-8x7b-32768

# Neo4j Configuration
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
```

## Usage

### Command Line Interface

1. Using OpenAI (ChatGPT):

```bash
python GraphRagChatgpt.py
```

2. Using Google Gemini:

```bash
python GraphRagGoogle.py
```

3. Using Groq:

```bash
python GraphRagGroq.py
```

4. Using Streamlit Interface:

```bash
streamlit run streamlit-app.py
```

### Programmatic Usage

```python
# Using ChatGPT version
from GraphRagChatgpt import initialize_and_load_pdf, ask_question

# Initialize system
chunks = initialize_and_load_pdf()

# Ask questions
answer = ask_question("What are the requirements for college admission?")
print(answer)
```

## Project Structure

- `GraphRagChatgpt.py`: OpenAI/ChatGPT implementation
- `GraphRagGoogle.py`: Google Gemini implementation
- `GraphRagGroq.py`: Groq implementation
- `utils.py`: Shared utilities and optimizations
- `config.py`: Configuration management
- `streamlit-app.py`: Web interface
- `requirements.txt`: Project dependencies

## Performance Optimizations

- **Text Processing**:

  - Recursive text splitting for better context preservation
  - Configurable chunk sizes and overlap

- **Vector Search**:

  - FAISS IVF indexing for efficient similarity search
  - Optimized batch processing
  - Configurable probe counts and fetch sizes

- **Knowledge Graph**:
  - Batched entity extraction and relationship creation
  - LLM-specific prompt optimization
  - Error handling and recovery

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the core framework
- Neo4j for graph database capabilities
- FAISS for efficient vector similarity search
- OpenAI, Google, and Groq for LLM APIs
