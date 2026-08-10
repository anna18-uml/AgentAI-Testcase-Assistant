from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config


embeddings = OllamaEmbeddings(
    model=config.OLLAMA_EMBEDDINGS_MODEL
)

requirements_store = Chroma(
    collection_name="requirements",
    persist_directory=config.CHROMA_DB_REQUIREMENTS_PATH,
    embedding_function=embeddings,
)

testcases_store = Chroma(
    collection_name="testcases",
    persist_directory=config.CHROMA_DB_TESTCASES_PATH,
    embedding_function=embeddings,
)


def search_requirements(query: str, k: int = 5) -> dict[str, Any]:
    """Retrieve requirement sections relevant to the goal."""

    documents = requirements_store.similarity_search(query, k=k)

    return {
        "tool": "search_requirements",
        "documents": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
            }
            for doc in documents
        ],
    }


def search_existing_tests(query: str, k: int = 10) -> dict[str, Any]:
    """Retrieve existing tests related to the requirement."""

    documents = testcases_store.similarity_search(query, k=k)

    return {
        "tool": "search_existing_tests",
        "documents": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
            }
            for doc in documents
        ],
    }


def check_duplicates(
    candidate_tests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Perform a simple similarity-based duplicate check."""

    results = []

    for candidate in candidate_tests:
        scenario = candidate.get("scenario", "")
        matches = testcases_store.similarity_search_with_score(
            scenario,
            k=1,
        )

        duplicate = False
        closest_match = None
        score = None

        if matches:
            document, score = matches[0]
            closest_match = document.page_content

            # Tune this threshold using your embedding model.
            duplicate = score < 0.20

        results.append(
            {
                "test_id": candidate.get("test_id"),
                "duplicate": duplicate,
                "similarity_distance": score,
                "closest_match": closest_match,
            }
        )

    return {
        "tool": "check_duplicates",
        "results": results,
    }


def export_test_cases(
    test_cases: list[dict[str, Any]],
    output_path: str = "generated_test_cases.csv",
) -> dict[str, Any]:
    """Export approved test cases to a CSV file."""

    if not test_cases:
        raise ValueError("No test cases were provided.")

    destination = Path(output_path)

    columns = [
        "test_id",
        "requirement_id",
        "scenario",
        "preconditions",
        "steps",
        "expected_result",
        "test_type",
        "source",
    ]

    with destination.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()

        for test in test_cases:
            row = dict(test)

            if isinstance(row.get("steps"), list):
                row["steps"] = " | ".join(row["steps"])

            writer.writerow(row)

    return {
        "tool": "export_test_cases",
        "path": str(destination.resolve()),
        "count": len(test_cases),
    }