# Source Code Analysis

A web application that allows users to analyze GitHub repositories using Large Language Model (LLM) operations. The app leverages OpenAI's GPT models and vector embeddings to answer questions about codebases, summarize code, and provide insights interactively.

## Features

- **Analyze any public GitHub repository** by URL
- **LLM-powered Q&A**: Ask questions about the codebase and get intelligent answers
- **Conversational memory**: Maintains chat context for multi-turn conversations
- **Embeddings-based retrieval**: Uses vector search to find relevant code/documentation


## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd source-code-analysis
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root with the following:
```
OPENAI_API_KEY=sk-<your-openai-api-key>
FLASK_SECRET_KEY=your-very-secret-key
```

### 5. Run the application
```bash
python app.py
```
The app will be available at [http://localhost:5000](http://localhost:5000)

