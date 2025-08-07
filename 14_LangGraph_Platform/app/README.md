## App package structure

This `app` package organizes LangGraph/LangChain agent graphs, shared state, model configuration, and tool integrations into focused modules. The goal is to keep each concern small and composable so you can mix and match graphs and capabilities without duplicating wiring.

### Layout

- `__init__.py`: Lightweight bootstrap that loads a local `.env` (for local dev) and exposes subpackages via `__all__`.
- `models.py`: Central place to construct chat LLM clients (e.g., OpenAI) with consistent defaults. Graphs import `get_chat_model()` instead of re-creating clients.
- `state.py`: Shared `AgentState` schema used by graphs. Uses `add_messages` to safely accumulate messages across steps.
- `tools.py`: Aggregates third-party tools (Tavily, Arxiv) and local tools (RAG) into a single tool belt for easy binding to models.
- `rag.py`: Minimal Retrieval-Augmented Generation pipeline. Loads PDFs from `RAG_DATA_DIR`, chunks, embeds, stores in in-memory Qdrant, and exposes a `retrieve_information` Tool.
- `graphs/`: Collection of agent graphs that orchestrate model calls, tool execution, and optional evaluation loops.
  - `simple_agent.py`: Smallest useful agent: model -> optional tools -> done.
  - `agent_with_helpfulness.py`: Adds a helpfulness evaluator loop that can route back to the agent or stop.

### Why this structure

- **Separation of concerns**: Models, state, tools, and graphs live in dedicated modules. Each can evolve independently (swap models, add tools, change routing) with minimal cross-coupling.
- **Reusability**: `get_tool_belt()` and `get_chat_model()` can be reused across multiple graphs; `AgentState` standardizes message passing.
- **Testability**: Small, focused modules are easier to test in isolation (e.g., unit test the RAG tool without the agent graphs).
- **Extensibility**: Add a new tool or graph by creating a new module without touching existing ones, then import/bind where needed.

### Environment variables

- `OPENAI_MODEL` or `OPENAI_CHAT_MODEL`: Controls which OpenAI chat model to use.
- `RAG_DATA_DIR`: Directory containing PDFs to index for the RAG tool (default: `data`).

### Typical usage

Graphs import from the shared modules:

```python
from app.models import get_chat_model
from app.tools import get_tool_belt
from app.state import AgentState
```

Then bind tools to the model and construct a `StateGraph` that routes between the agent node and an `ToolNode` for tool execution.


