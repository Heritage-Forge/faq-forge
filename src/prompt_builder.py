def build_prompt(retrieved: list, user_query: str) -> str:
    """
    Build a RAG-style prompt by concatenating retrieved FAQ answers with the user query.
    
    Args:
        retrieved (list): List of retrieved FAQ items (each with keys "answer", etc.)
        user_query (str): The original user query.
        
    Returns:
        str: Constructed prompt.
    """
    # Create a context string from retrieved answers
    context = "\n".join(
        [f"Context {i+1}: {item['answer']}" for i, item in enumerate(retrieved)]
    )
    prompt = (
        f"Use the following context to answer the question:\n\n"
        f"{context}\n\n"
        f"User question: {user_query}\n"
        f"Answer:"
    )
    return prompt
