def extract_messages(response):
    """Extract text content from messages for JSON serialization"""
    return [{"type": msg.__class__.__name__, "content": msg.content} for msg in response["messages"]]
