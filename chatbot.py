from langchain_ollama.chat_models import ChatOllama

import config
from knowledge import search_knowledge, documents_as_text


llm = ChatOllama(
    model=config.OLLAMA_GENERATION_MODEL,
    temperature=0.2,
)


def ask(question: str):
    """
    Normal AI assistant.

    Simple greetings and thanks are handled directly so they do not
    trigger vector search. Technical questions use the shared knowledge
    base and answer only from retrieved documents.
    """

    normalized = question.lower().strip()

    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }

    thanks = {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
    }

    # Handle greetings directly.
    if normalized in greetings:
        return {
            "answer": (
                "Hello! How can I assist you today? "
                "You can ask me questions about the documents "
                "available in the TechOmni knowledge base."
            ),
            "documents": [],
        }

    # Handle thanks directly.
    if normalized in thanks:
        return {
            "answer": (
                "You're welcome! Let me know if you need anything else."
            ),
            "documents": [],
        }

    # Normal RAG flow.
    docs = search_knowledge(
        question,
        k=6,
    )

    context = documents_as_text(
        docs
    )

    prompt = f"""
You are TechOmni, a technical assistant.

Answer the user's question using ONLY the provided knowledge.

If the retrieved knowledge does not contain enough information
to answer the question, respond:

"I could not find enough information in the available documents."

Do not make up an answer.

KNOWLEDGE:
{context}

USER QUESTION:
{question}
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content,
        "documents": docs,
    }
