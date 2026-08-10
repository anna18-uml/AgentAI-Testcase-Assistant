from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import *
from uuid import uuid4

embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
feedbackstore = Chroma(collection_name="feedback", persist_directory=FEEDBACK_DB_PATH, embedding_function=embeddings)

feedbackstore_4007 = Chroma(collection_name="4007", persist_directory=FEEDBACK_DB_4007_PATH, embedding_function=embeddings)
feedbackstore_4100 = Chroma(collection_name="4100", persist_directory=FEEDBACK_DB_4100_PATH, embedding_function=embeddings)
feedbackstore_4100_reg = Chroma(collection_name="4100_reg", persist_directory=FEEDBACK_DB_4100_REG_PATH, embedding_function=embeddings)

temp_ids = []
def update_feedback_db(feedbackstore, message):
    """Update the feedback database with the new message."""
    response = message.get("content", None)
    if response is None:
        print("Response is missing.")
        return
    query = response.get("input", None)
    feedback = message.get("feedback", None)
    contexts = response.get("context", None)
    doc = None

    if not all([query, contexts]):
        print("Query or feedback is missing.")
        return
    
    context_ids = [context.metadata.get("unique_id", "Unknown") for context in contexts]
    # Perform a similarity search with scores
    results = feedbackstore.similarity_search_with_score(query=query, k=1)  # k=1 for the top match
    
    if results:
        doc, score = results[0]  # Get the top match and its score
        if score == 0.0:
            feedback_score = 0
            if feedback == 0:
                feedback_score = -1
            elif feedback == 1:
                feedback_score = 1
            updated_feedback_score = max(-100, min(100, doc.metadata.get("feedback_score", 0) + feedback_score))
            updated_context_ids = ",".join(list(set(doc.metadata.get("context_ids", str).split(",")) | set(context_ids)))
            updated_doc = Document(page_content=doc.page_content, id=doc.id, metadata={"feedback_score": updated_feedback_score,"context_ids": updated_context_ids})
            feedbackstore.update_documents(ids=[doc.id], documents=[updated_doc])
        else:
            doc = None
    
    if doc == None:
        # If no match is found, create a new document
        feedback_score = 0
        if feedback == 0:
            feedback_score = -1
        elif feedback == 1:
            feedback_score = 1
        docid = str(uuid4())
        temp_ids.append(docid)
        doc = Document(page_content=query, id=docid, metadata={"feedback_score": feedback_score,"context_ids": ",".join(context_ids)})
        feedbackstore.add_documents([doc])
            
    
def boost_score_for_context_ids(feedbackstore, message):
    """Boost the score for each matching context ID in the passed message."""

    query = message.get("input", None)
    context_ids = [context.metadata.get("unique_id", "Unknown") for context in message.get("context", [])]
    
    if not query and not context_ids:
        print("Query or context IDs are missing.")
        return {}
    
    score_boost = {}
    for context_id in context_ids:
        score_boost[context_id] = 0  # Initialize boost factor for each context ID

    # Perform a similarity search with scores
    results = feedbackstore.similarity_search_with_score(query, k=10)  # Retrieve top 10 matches

    for doc, score in results:
        if score < 0.1:  # Check for high similarity (low score)
            # Check if any context ID matches
            existing_context_ids = list(doc.metadata.get("context_ids", str).split(","))
            matching_ids = set(existing_context_ids) & set(context_ids)
            feedback_score = doc.metadata.get("feedback_score", 0)

            for context_id in matching_ids:
                # Boost the score for each matching context ID
                score_boost[context_id] = max(-100, min(100, score_boost[context_id] + feedback_score))

    return score_boost
