# Graph RAG System

A Retrieval-Augmented Generation (RAG) system that combines graph-based knowledge representation with language models for question answering.

## Features

- Graph-based knowledge representation using Neo4j
- Hybrid retrieval combining structured and unstructured data
- Conversational memory for follow-up questions
- Wikipedia data integration
- Graph visualization capabilities

## Requirements

- Python 3.8+
- Neo4j Database
- OpenAI API key
- Required Python packages (see requirements.txt)

## Setup

1. Set up environment variables:

   ```bash
   export OPENAI_API_KEY="your-api-key"
   export NEO4J_URI="your-neo4j-uri"
   export NEO4J_USERNAME="your-username"
   export NEO4J_PASSWORD="your-password"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:

```python
from GraphRag import ask_question

# Ask a single question
answer = ask_question("Who was Elizabeth I?")
print(answer)

# Ask follow-up questions
chat_history = [("Who was Elizabeth I?", answer)]
follow_up = ask_question("When was she born?", chat_history)
print(follow_up)

# Visualize the graph
from GraphRag import show_graph
widget = show_graph()
display(widget)  # In Jupyter notebook
```

## Testing

Run the basic tests:

```bash
python GraphRag.py
```

Run comprehensive tests:

```bash
python test.py
```

## Components

- `GraphRag.py`: Core implementation including graph initialization, retrieval, and QA functionality
- `test.py`: Comprehensive test suite with various test scenarios
- `requirements.txt`: Required dependencies

## Architecture

1. **Data Ingestion**

   - Wikipedia article loading
   - Text splitting and processing
   - Graph transformation

2. **Retrieval**

   - Structured graph-based retrieval
   - Vector similarity search
   - Hybrid retrieval combination

3. **Question Answering**
   - Question processing
   - Context retrieval
   - Answer generation

## Example Queries

The system can handle various types of questions:

- Basic facts: "Who was Elizabeth I?"
- Relationships: "Who were Elizabeth I's parents?"
- Complex queries: "How did Elizabeth I's relationship with Spain evolve?"
- Follow-up questions: "When did she become queen?"

## Error Handling

The system includes error handling for:

- Empty or invalid questions
- Connection issues
- API failures
- Invalid chat history

## Limitations

- Currently focused on Elizabeth I's historical context
- Requires active internet connection
- API key and Neo4j database required
- May have rate limiting based on API usage

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License

## Acknowledgments

- OpenAI for GPT models
- Neo4j for graph database
- LangChain for the framework
- Wikipedia for the source data

python -m venv venv
venv\Scripts\activate
