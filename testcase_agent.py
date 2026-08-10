from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_ollama.chat_models import ChatOllama

import config
from knowledge import search_knowledge, documents_as_text

@dataclass
class AgentState:
    goal: str
    activity: list[str] = field(default_factory=list)
    generated_tests: list[dict[str, Any]] = field(default_factory=list)
    final_summary: str = ""
    completed: bool = False

class TestCaseAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_GENERATION_MODEL,
            temperature=0.1,
            format="json",
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("Model did not return valid JSON.")
            return json.loads(match.group(0))

    def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)

        state.activity.append(
            "Step 1: Controller received the test-generation goal."
        )

        planning_prompt = f"""
You are a test engineering agent.

Goal:
{goal}

Decide what information should be searched in the shared engineering
knowledge base before generating test cases.

Return ONLY valid JSON:

{{
  "search_query": "concise search query",
  "reason": "why the search is needed"
}}
"""

        plan = self._parse_json(self.llm.invoke(planning_prompt).content)
        query = plan.get("search_query", "").strip() or goal

        state.activity.append(
            "Step 2: LLM planned the knowledge search — "
            + plan.get("reason", "retrieve relevant engineering information")
        )

        docs = search_knowledge(query, k=8)

        state.activity.append(
            f"Step 3: Controller executed search_knowledge() and retrieved "
            f"{len(docs)} relevant document sections."
        )

        if not docs:
            state.final_summary = (
                "No relevant information was found in the shared knowledge folder."
            )
            return state

        context = documents_as_text(docs)

        generation_prompt = f"""
You are a software test engineer.

GOAL:
{goal}

RETRIEVED KNOWLEDGE:
{context}

Generate test cases supported by the retrieved knowledge.

Rules:
- Do not invent thresholds, timings, or expected behavior.
- Create positive, negative, boundary, and recovery cases only when supported.
- Include the source file for each case.
- Return ONLY valid JSON.

Format:

{{
  "summary": "short summary",
  "test_cases": [
    {{
      "test_id": "TC-001",
      "scenario": "scenario",
      "preconditions": "preconditions",
      "steps": ["step 1", "step 2"],
      "expected_result": "expected result",
      "test_type": "positive",
      "source": "source file"
    }}
  ]
}}
"""

        output = self._parse_json(self.llm.invoke(generation_prompt).content)

        state.activity.append(
            "Step 4: LLM analyzed the retrieved knowledge and generated test cases."
        )

        state.generated_tests = output.get("test_cases", [])
        state.final_summary = output.get(
            "summary",
            f"Generated {len(state.generated_tests)} test cases.",
        )

        state.activity.append(
            "Step 5: Controller returned the test cases for human review."
        )

        state.completed = True
        return state
