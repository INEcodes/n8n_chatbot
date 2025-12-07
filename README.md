# n8n_chatbot

A powerful RAG (Retrieval-Augmented Generation) chatbot for n8n pre-defined and available templates, powered by Google's Gemini API and Pinecone vector database.

## 🌐 Live Demo

Try the chatbot here: [https://n8nchatbot.streamlit.app/](https://n8nchatbot.streamlit.app/)

## 📋 Overview

This is a web-based chatbot application built with Streamlit that combines:
- **Vector Search**: Pinecone for efficient document retrieval
- **Embeddings**: Google Gemini's embedding model for semantic search
- **LLM**: Google Gemini 2.5 Flash for intelligent response generation

The chatbot retrieves relevant documents from your knowledge base and generates contextual answers based on the retrieved information.

## ✨ Features

- 🤖 Intelligent question-answering with context awareness
- 📚 Document retrieval from Pinecone vector database
- 🔍 Semantic search using Gemini embeddings
- 💬 Interactive chat interface with message history
- 📖 Source citations with relevance scores
- ⚙️ Customizable settings (document retrieval count, response creativity)
- 🗑️ Clear chat history functionality
- 📊 Confidence scores for retrieved documents

## 🛠️ Prerequisites

Before running this application, ensure you have:

1. **Python 3.8+** installed
2. **API Keys**:
   - Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))
   - Pinecone API Key (from [Pinecone Console](https://app.pinecone.io/))
3. **Pinecone Index**: A Pinecone index named `project` with documents already indexed

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/n8n_chatbot.git
cd n8n_chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

**Important**: Add `.env` to `.gitignore` (already done) to keep your API keys secure.

## 🚀 Running the Application

### Local Development

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Production Deployment

The app is deployed on Streamlit Cloud at: [https://n8nchatbot.streamlit.app/](https://n8nchatbot.streamlit.app/)

To deploy your own version:
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and connect your GitHub repository
4. Set environment variables in the Streamlit Cloud dashboard

## 📖 How to Use

1. **Ask a Question**: Type your question in the input box at the bottom
2. **Wait for Processing**: The chatbot will:
   - Search for relevant documents in Pinecone
   - Generate an answer based on the retrieved context
3. **View Sources**: Click "📚 View Sources" to see:
   - Retrieved documents with titles
   - Relevance scores (0-1 scale)
   - Source links if available
4. **Adjust Settings** (Sidebar):
   - **Number of documents**: Retrieve 1-10 documents (default: 3)
   - **Temperature**: Control response creativity (0.0-1.0, default: 0.7)
5. **Clear History**: Use the "🗑️ Clear Chat History" button to reset the conversation

## 🏗️ Project Structure

```
n8n_chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in git)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## 🔧 Key Components

### Configuration
- **Pinecone Index**: `project`
- **Embedding Model**: `models/text-embedding-004` (Gemini)
- **Chat Model**: `gemini-2.5-flash`
- **Default Namespace**: `_default_`

### Main Functions

1. **`create_gemini_embedding(text)`**: Creates embeddings for semantic search
2. **`retrieve_relevant_documents(query, index, top_k)`**: Retrieves relevant documents from Pinecone
3. **`generate_response(query, retrieved_docs)`**: Generates contextual responses using Gemini
4. **`init_chatbot()`**: Initializes Pinecone connection with caching

## 🔐 Security

- API keys are stored in `.env` and never committed to git
- Environment variables are used for sensitive configuration
- Streamlit Cloud supports secure environment variable management

## 📝 Requirements

```
streamlit>=1.28.0
python-dotenv>=1.0.0
pinecone-client>=3.0.0
google-generativeai>=0.3.0
```

## 🐛 Troubleshooting

### Missing API Keys Error
```
❌ Missing required API keys in environment variables
```
**Solution**: Ensure `GOOGLE_API_KEY` and `PINECONE_API_KEY` are set in your `.env` file

### Failed to Connect to Pinecone
```
❌ Failed to connect to Pinecone. Please check your API keys.
```
**Solution**: 
- Verify Pinecone API key is correct
- Ensure the index name `project` exists in Pinecone
- Check your network connection

### No Documents Retrieved
**Solution**:
- Verify your Pinecone index has documents indexed
- Check document metadata contains required fields (`text`, `title`, or `description`)
- Try different search queries

## 🚀 Next Steps

- Customize the system prompt in `generate_response()` for specific use cases
- Add document upload functionality
- Implement user authentication
- Add support for multiple Pinecone indexes
- Create advanced filtering options

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Support

For issues or questions, please open an issue on GitHub or contact the maintainers.

---

**Live Demo**: [https://n8nchatbot.streamlit.app/](https://n8nchatbot.streamlit.app/)
