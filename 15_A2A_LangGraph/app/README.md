# 🤖 LangGraph Agent Implementation

This directory contains the complete implementation of a LangGraph agent with A2A protocol and helpfulness evaluation. This README provides detailed technical documentation for understanding and extending the codebase.

## 📁 File Structure

```
📦 app/
├── 📄 __init__.py                           # Package initialization
├── 📄 __main__.py                           # Entry point for A2A server
├── 📄 agent.py                              # Core agent implementation with ResponseFormat
├── 📄 agent_executor.py                     # A2A protocol executor and server setup
├── 📄 agent_graph_with_helpfulness.py      # LangGraph with helpfulness evaluation
├── 📄 rag.py                                # RAG implementation with Qdrant vectorstore
├── 📄 tools.py                              # Tool belt configuration (Tavily, ArXiv, RAG)
├── 📄 test_client.py                        # Test client for the agent API
└── 📄 README.md                             # This file
```

## 🔧 Core Components

### 1. `agent_graph_with_helpfulness.py`

**Purpose**: Implements the main LangGraph with helpfulness evaluation loop.

**Key Components**:
- `AgentState`: TypedDict defining the state schema with message history
- `build_model_with_tools()`: Binds tools to the language model
- `call_model()`: Main agent node that processes messages and generates responses
- `route_to_action_or_helpfulness()`: Router deciding between tool execution and evaluation
- `helpfulness_node()`: A2A evaluation node that assesses response quality
- `helpfulness_decision()`: Decision node for continuing or terminating the loop

**Graph Structure**:
```python
graph.add_node("agent", _call_model)           # Main LLM + tools
graph.add_node("action", tool_node)            # Tool execution
graph.add_node("helpfulness", _helpfulness_node)  # A2A evaluation
```

**Flow Logic**:
1. Start at `agent` node
2. If tool calls needed → `action` node → back to `agent`
3. If no tool calls → `helpfulness` node
4. Helpfulness evaluation: Y (end) or N (continue, max 10 loops)

### 2. `agent.py`

**Purpose**: Defines the main Agent class with streaming capabilities and response formatting.

**Key Features**:
- `ResponseFormat`: Pydantic model for structured responses
- OpenAI model integration
- Streaming interface with real-time updates
- A2A protocol compliance with status tracking

**Response States**:
- `input_required`: User needs to provide more information
- `completed`: Request successfully fulfilled
- `error`: Error occurred during processing

### 3. `tools.py`

**Purpose**: Assembles the tool belt available to agents.

**Available Tools**:
```python
def get_tool_belt() -> List:
    tavily_tool = TavilySearchResults(max_results=5)  # Web search
    return [
        tavily_tool,           # Real-time web search
        ArxivQueryRun(),       # Academic paper search
        retrieve_information   # RAG document retrieval
    ]
```

### 4. `rag.py`

**Purpose**: Complete RAG (Retrieval-Augmented Generation) implementation.

**Architecture**:
- **Document Loading**: Recursively loads PDFs from `RAG_DATA_DIR`
- **Text Splitting**: Token-aware chunking with `RecursiveCharacterTextSplitter`
- **Embeddings**: OpenAI embeddings for vector representation
- **Vector Store**: In-memory Qdrant for similarity search
- **RAG Graph**: Two-node LangGraph (retrieve → generate)

**Token-Aware Chunking**:
```python
def _tiktoken_len(text: str) -> int:
    """Return token length using tiktoken for accurate chunk sizing."""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)
```

### 5. `agent_executor.py`

**Purpose**: A2A protocol server implementation using FastAPI.

**Key Features**:
- RESTful API endpoints for agent interaction
- Streaming response support
- Context management for multi-turn conversations
- Error handling and protocol compliance

### 6. `test_client.py`

**Purpose**: Test client for interacting with the agent API.

**Usage**:
```bash
uv run python app/test_client.py
```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
TOOL_LLM_URL=https://api.openai.com/v1
TOOL_LLM_NAME=gpt-4o-mini

# Tool Configuration
TAVILY_API_KEY=your_tavily_api_key

# RAG Configuration
RAG_DATA_DIR=data
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### Document Setup for RAG

```bash
# Create data directory
mkdir -p data

# Add PDF documents
cp /path/to/your/documents/*.pdf data/
```

## 🚀 Running the Agent

### Local Development

```bash
# Start the A2A server
uv run python -m app

# Or with custom host/port
uv run python -m app --host 0.0.0.0 --port 8080
```

### LangGraph Server

```bash
# Start LangGraph development server
uv run langgraph dev

# Server will be available at:
# API: http://localhost:2024
# Studio: https://smith.langchain.com/studio?baseUrl=http://localhost:2024
```

## 🧪 Testing

### Using the Test Client

```bash
uv run python app/test_client.py
```

### Direct API Calls

```bash
# Start a conversation
curl -X POST http://localhost:2024/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "messages": [{
      "role": "user",
      "content": "Find recent papers about transformers"
    }]
  }'

# Continue conversation
curl -X POST http://localhost:2024/v1/tasks/{task_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Summarize the key findings"}'
```

### Example Queries by Tool Type

| Query Type | Example | Expected Tool |
|------------|---------|---------------|
| **Web Search** | "What are the latest AI developments in 2024?" | Tavily Search |
| **Academic** | "Find papers on multimodal transformers" | ArXiv Search |
| **Documents** | "What do the policy documents say about requirements?" | RAG Retrieval |
| **Multi-Tool** | "Compare recent research with our internal guidelines" | ArXiv + RAG |

## 🔄 A2A Protocol Deep Dive

### Helpfulness Evaluation

The helpfulness node implements sophisticated response evaluation:

```python
def helpfulness_node(state: Dict[str, Any], model) -> Dict[str, Any]:
    # Extract initial query and final response
    initial_query = state["messages"][0]
    final_response = state["messages"][-1]
    
    # Evaluation prompt template
    prompt_template = """
    Given an initial query and a final response, determine if the final response is extremely helpful or not. 
    A helpful response should:
    - Provide accurate and relevant information
    - Be complete and address the user's specific need
    - Use appropriate tools when necessary
    
    Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.
    """
    
    # Evaluate and return decision
    decision = "Y" if "Y" in helpfulness_response else "N"
    return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}
```

### Loop Protection

The system prevents infinite loops through multiple mechanisms:

1. **Iteration Counter**: Tracks message count (max 10 iterations)
2. **Hard Stop**: Returns `HELPFULNESS:END` when limit exceeded
3. **Decision Router**: Routes to END state on termination conditions

### State Management

The `AgentState` TypedDict manages conversation state:

```python
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]  # Conversation history
    structured_response: Any                 # Formatted response data
```

## 🎯 Customization Guide

### Adding New Tools

1. **Create the tool** (in `tools.py`):
```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Description of what your tool does"""
    # Implementation here
    return result
```

2. **Add to tool belt**:
```python
def get_tool_belt() -> List:
    return [
        TavilySearchResults(max_results=5),
        ArxivQueryRun(),
        retrieve_information,
        my_custom_tool  # Add your tool here
    ]
```

### Customizing Helpfulness Evaluation

Modify the evaluation criteria in `agent_graph_with_helpfulness.py`:

```python
prompt_template = """
Given an initial query and a final response, determine if the final response is extremely helpful or not. 
A helpful response should:
- [Add your custom criteria here]
- Be factually accurate
- Include relevant sources
- Address all parts of the question

Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.
"""
```

### Extending RAG Capabilities

Modify `rag.py` to add new document types or processing:

```python
# Add new document loaders
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyMuPDFLoader,
    TextLoader,      # Add text files
    CSVLoader        # Add CSV files
)

# Modify loader configuration
def _build_rag_graph(data_dir: str):
    # Load multiple file types
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    # ... combine loaders
```

## 🐛 Troubleshooting

### Quick Diagnostics

```bash
# Check environment configuration
uv run python check_env.py

# Test tool availability
uv run python -c "from app.tools import get_tool_belt; print([tool.name for tool in get_tool_belt()])"

# Test RAG loading
uv run python -c "from app.rag import get_rag_graph; rag = get_rag_graph(); print('RAG loaded successfully')"
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Internal Error (-32603)** | Invalid model names or missing API keys | Check `TOOL_LLM_NAME` and API keys in `.env` |
| **Tool Call Failures** | Missing API keys for tools | Verify `TAVILY_API_KEY` and `OPENAI_API_KEY` |
| **RAG Errors** | No documents in data directory or missing OpenAI key | Add PDFs to `data/` and set `OPENAI_API_KEY` |
| **Timeout Errors** | LLM responses taking too long | Helpfulness evaluation can take 10-30s, adjust timeouts |
| **Import Errors** | Missing dependencies | Run `uv sync` or `pip install -e .` |

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LANGCHAIN_VERBOSE=true
```

### Memory Issues

For large document collections:

```python
# Reduce chunk size in rag.py
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,    # Reduce from 1000
    chunk_overlap=50,  # Reduce from 100
    # ...
)
```

## 📊 Performance Optimization

### Response Time Optimization

1. **Reduce Tool Results**: Limit `max_results` in tool configurations
2. **Optimize Chunk Size**: Balance retrieval quality vs. speed
3. **Cache Embeddings**: Implement vector store persistence
4. **Async Operations**: Use async tool implementations where possible

### Memory Optimization

1. **Document Chunking**: Optimize chunk size and overlap
2. **Vector Store**: Consider persistent storage for large collections
3. **Model Selection**: Choose appropriate model sizes for your use case

## 🔮 Advanced Features

### Multi-Agent Communication

Extend the A2A protocol for agent-to-agent communication:

```python
# Example: Specialized agents for different domains
class SpecializedAgent(Agent):
    def __init__(self, domain: str):
        super().__init__()
        self.domain = domain
        # Customize tools and instructions for domain
```

### Custom Evaluation Metrics

Implement additional evaluation criteria:

```python
def enhanced_helpfulness_node(state, model):
    # Evaluate multiple dimensions:
    # - Factual accuracy
    # - Completeness
    # - Relevance
    # - Source quality
    # Return structured evaluation
```

### Integration with External Services

Connect to external APIs and services:

```python
@tool
def external_api_tool(query: str) -> str:
    """Tool that calls external API"""
    # Implement API integration
    return api_response
```

---

This implementation provides a solid foundation for understanding and extending LangGraph agents with A2A protocol compliance. The modular architecture makes it easy to customize tools, evaluation criteria, and response formats while maintaining the core helpfulness evaluation loop.
