import os
import streamlit as st

import chatbot
from testcase_agent import TestCaseAgent

st.set_page_config(page_title="Agentic AI Demo", page_icon="🤖")

st.title("Agentic AI Demo")
st.caption("Normal AI and Agentic AI use the same shared knowledge folder.")

normal_tab, agent_tab = st.tabs(
    ["Normal AI Assistant", "Agentic Test Case Assistant"]
)

with normal_tab:
    st.subheader("Normal AI Assistant")
    question = st.text_area(
        "Question",
        placeholder="What does the documentation say about battery trouble?",
    )

    if st.button("Ask Normal AI", disabled=not question.strip()):
        try:
            with st.spinner("Searching knowledge..."):
                result = chatbot.ask(question)

            st.markdown("### Answer")
            st.write(result["answer"])

            with st.expander("Retrieved sources"):
                for doc in result["documents"]:
                    source = os.path.basename(
                        doc.metadata.get("source", "Unknown")
                    )
                    st.markdown(f"**{source}**")
                    st.write(doc.page_content[:800])
                    st.divider()

        except Exception as exc:
            st.error(f"Normal AI failed: {exc}")

@st.cache_resource
def load_agent():
    return TestCaseAgent()

agent = load_agent()

with agent_tab:
    st.subheader("Agentic Test Case Assistant")

    goal = st.text_area(
        "Test-generation goal",
        placeholder=(
            "Generate test cases for battery trouble detection. "
            "Include positive, negative, boundary, and recovery cases "
            "where supported by the documentation."
        ),
        key="agent_goal",
    )

    if st.button("Start Agent", type="primary", disabled=not goal.strip()):
        try:
            with st.spinner("Agent is working..."):
                result = agent.run(goal)
            st.session_state["agent_result"] = result
        except Exception as exc:
            st.error(f"Agent failed: {exc}")

    result = st.session_state.get("agent_result")

    if result:
        st.markdown("### Agent Activity")
        for item in result.activity:
            st.write(item)

        st.markdown("### Summary")
        st.info(result.final_summary)

        if result.generated_tests:
            st.markdown("### Generated Test Cases")
            st.dataframe(result.generated_tests, use_container_width=True)

            st.warning(
                "Generated test cases require engineer review before use."
            )
