# AgentAI-Testcase-Assistant

## Prerequisites
1. [Download and install Ollama](https://ollama.com/download).
2. Verify Mistral is running by executing the following command in the terminal:
    ```bash
    ollama run mistral
    ```

## Setup Instructions

1. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
2. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
3. From the root folder, install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Run Instructions

1. Initialize the vector database:
    ```bash
    python vectordb.py
    ```
2. Launch the application:
    ```bash
    streamlit run techomni.py
    ```

## Sample Questions
- **Question:** How to edit ES Net Network Interface card?
- **Question:** What are the computer requirements for 4007 ES Panel Programmer?
