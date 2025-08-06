## 🏗️ Activity #2: MCP Client/Server Integration Approaches

### Method 1: LangChain MCP Adapters

See `method_one.py`

**Approach**: Direct integration using `langchain_mcp_adapters` with a ReAct agent

**Architecture**: Simple client-server connection via stdio

**Use Cases**:
- Rapid prototyping with MCP servers
- Single-query workflows
- Direct tool access without state management

### Method 2: LangGraph State Management

See `method_two.py`

**Approach**: Custom state graph with MultiServerMCPClient and structured output handling

**Architecture**: StateGraph with conditional edges, tool nodes, and persistent file storage

**Use Cases**:
- Complex multi-step workflows
- Batch processing multiple queries
- Structured data persistence and analysis
- Production applications requiring state management

### Supporting Utilities:

- `extract_messages.py`: Message serialization for JSON storage
- `file_saver.py`: Timestamped file output with directory organization

Both approaches demonstrate MCP server integration but serve different complexity levels and operational requirements.
