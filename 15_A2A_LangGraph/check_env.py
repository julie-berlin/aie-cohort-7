#!/usr/bin/env python3
"""Quick script to check environment configuration."""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("🔍 Checking Environment Configuration")
print("=" * 50)

# Check API keys
api_keys = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'TAVILY_API_KEY': os.getenv('TAVILY_API_KEY'),
}

print("\nAPI Keys Status:")
for key, value in api_keys.items():
    if value:
        # Show only first/last 4 chars for security
        if len(value) > 8:
            masked = f"{value[:4]}...{value[-4:]}"
        else:
            masked = "****"
        print(f"  ✅ {key}: {masked}")
    else:
        print(f"  ❌ {key}: Not set")

# Check LLM configuration
if api_keys['OPENAI_API_KEY']:
    print("\n✅ OpenAI configuration looks good")
    print(f"  - LLM URL: {os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1')}")
    print(f"  - LLM Name: {os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini')}")
else:
    print("\n❌ OpenAI API key missing")

# Check RAG configuration
print(f"\nRAG Configuration:")
print(f"  - Data Directory: {os.getenv('RAG_DATA_DIR', 'data')}")
print(f"  - Chat Model: {os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}")

if not api_keys['OPENAI_API_KEY']:
    print("  ⚠️  Note: OpenAI API key required for RAG embeddings and main LLM")

# Check if data directory exists
data_dir = os.getenv('RAG_DATA_DIR', 'data')
if os.path.exists(data_dir):
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    print(f"  - PDF files found: {len(pdf_files)}")
else:
    print(f"  - ❌ Data directory '{data_dir}' does not exist")

print("\n" + "=" * 50)
print("\n💡 Tip: Run 'python setup.py' to configure your environment")
