# Git LLAMA 🦙

## Overview

Git LLAMA is a powerful Streamlit-based application designed to analyze and interact with GitHub repositories. Utilizing advanced language models such as LLaMA and tools like FAISS, this tool offers a seamless way to explore repository structures, retrieve specific insights, and answer queries based on repository content.

---

## Features

- **Repository Analysis**: Analyze and display the structure, summary, and content of any GitHub repository.
- **Interactive Chat Interface**: Ask questions about the repository's content and implementation, with responses tailored to the specific structure and files.
- **Enhanced Retrieval**: Combines FAISS vector storage with LLaMA embeddings for precise and efficient retrieval of repository information.
- **Customizable UI**: Includes a clean and intuitive interface, collapsible sections, and a sidebar for managing repositories.
- **Session Management**: Maintains chat history and repository data across sessions with optimized caching.

---

## Prerequisites

- Python 3.9+
- Environment variables configured via `.env` file.

### Python Packages

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

#### Key Libraries Used:

- `streamlit`: For building the interactive web app.
- `langchain`: For chain creation, embeddings, and retrieval.
- `faiss-cpu`: For vector storage and similarity search.
- `dotenv`: For loading environment variables.
- `gitingest`: Custom GitHub repository processing tool.

---

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/git-llama
cd git-llama
```

### 2. Set Up Environment Variables

Create a `.env` file with the following variables:

```env
GITHUB_TOKEN=your_github_personal_access_token
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Interact with the Interface

- **Load Repository**: Enter the GitHub repository URL in the sidebar and click "Load Repository".
- **Ask Questions**: Type your query in the input box and click "Submit Query".
- **View Repository Structure**: Expand the "Repository Structure" section to see the tree view of the repository.

---

## Application Workflow

1. **Repository Loading**:

   - Accepts a GitHub repository URL.
   - Processes repository structure, summary, and content using `gitingest`.
   - Splits content into manageable chunks with `RecursiveCharacterTextSplitter`.

2. **Retrieval Chain Creation**:

   - Embeddings generated using `OllamaEmbeddings`.
   - Stores embeddings in a FAISS vector store.
   - Constructs a custom retrieval chain with enhanced prompts.

3. **Query Handling**:
   - User queries are processed in real-time.
   - Combines chat history, repository context, and user query for accurate answers.

---

## Code Highlights

### Streamlit Customization

- **Page Configuration**:
  ```python
  st.set_page_config(page_title="Git LLAMA", page_icon="🦙", layout="wide")
  ```
- **Enhanced UI**: Includes custom CSS for better chat and repository display.

### Retrieval Chain

- **Prompt Template**:
  ```python
  You are an AI assistant specialized in analyzing GitHub repositories.
  Repository structure:
  {context}
  ---------------------
  Previous conversation:
  {chat_history}
  ---------------------
  Query: {input}
  Answer:
  ```
- **Chain Creation**:
  ```python
  create_retrieval_chain(retriever, document_chain)
  ```

---

## Acknowledgment of Similar Projects

This project was inspired by a similar RAG (Retrieval-Augmented Generation) application developed by Akshay Prashar using `llamaindex`. You can explore Akshay's implementation here: [AI Engineering Hub - GitHub RAG](https://github.com/patchy631/ai-engineering-hub/tree/main/github-rag).

Git LLAMA was created using `langchain` to test my knowledge and capabilities with this framework.

---

<!--
## Screenshots

1. **Main Interface**:
   ![Main Interface](images/main_interface.png)

2. **Repository Structure**:
   ![Repository Structure](images/repo_structure.png)

3. **Query Response**:
   ![Query Response](images/query_response.png)

--- -->

## Future Enhancements

- **Multi-repository Support**: Allow users to analyze multiple repositories simultaneously.
- **Advanced Visualizations**: Add charts and graphs for better repository insights.
- **Integration with CI/CD**: Provide suggestions for improving workflows based on repository content.
- **Deployment Options**: Enable hosting on platforms like Heroku or AWS.

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Submit a pull request with a detailed description.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- **Streamlit**: For providing an amazing framework for building interactive web apps.
- **LangChain**: For enabling seamless integrations with language models and retrieval tools.
- **FAISS**: For efficient vector similarity search.
