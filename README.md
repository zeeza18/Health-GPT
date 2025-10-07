# 🏥 Health-GPT

An end-to-end medical chatbot powered by Llama 2, LangChain, and Pinecone vector database. Health-GPT provides accurate, context-aware responses to health-related queries by leveraging Retrieval-Augmented Generation (RAG) to ground answers in trusted medical documents.

## 📋 Overview

Health-GPT is a Flask-based web application that combines the power of large language models with domain-specific medical knowledge retrieval. By using RAG architecture, the chatbot retrieves relevant medical information from a vector database before generating responses, ensuring accuracy and reducing hallucinations common in pure LLM approaches.

### Key Features

- **🤖 Local LLM**: Runs Llama 2 7B model locally for privacy and control
- **📚 RAG Architecture**: Retrieves context from medical documents before answering
- **🔍 Semantic Search**: Uses sentence transformers for intelligent document retrieval
- **💬 Interactive Chat UI**: Clean, responsive web interface with real-time messaging
- **🔒 Privacy-Focused**: All processing happens locally - no data sent to external APIs
- **📖 Medical Knowledge Base**: Pre-indexed medical documents for accurate health information

## 🛠️ Tech Stack

### Backend
- **Flask** - Web framework
- **LangChain** - LLM orchestration and RAG pipeline
- **Llama 2 7B (GGML)** - Quantized language model for efficient inference
- **CTransformers** - Fast inference engine for GGML models

### Vector Database & Embeddings
- **Pinecone** - Cloud-based vector database for semantic search
- **Sentence Transformers** - HuggingFace embeddings for document encoding
- **PyPDF** - PDF processing and text extraction

### Frontend
- **HTML/CSS/JavaScript** - Clean chat interface
- **Bootstrap 4** - Responsive design
- **jQuery** - AJAX communication with backend

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (for running Llama 2 locally)
- Pinecone API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/zeeza18/Health-GPT.git
cd Health-GPT
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
ctransformers
sentence-transformers
pinecone-client
langchain-community
flask
pypdf
python-dotenv
```

### Step 4: Download Llama 2 Model

Download the quantized Llama 2 model:
- **Model**: `llama-2-7b-chat.ggmlv3.q4_0.bin`
- **Source**: [HuggingFace - TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

Create a `model/` directory and place the downloaded model there:
```bash
mkdir model
# Move llama-2-7b-chat.ggmlv3.q4_0.bin to model/
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
```

Get your Pinecone API key from [Pinecone Console](https://app.pinecone.io/)

### Step 6: Prepare Your Medical Documents

1. Create a `data/` directory
2. Add your medical PDFs to this directory

```bash
mkdir data
# Add your medical PDF files to data/
```

### Step 7: Index Documents to Pinecone

Run the indexing script to process PDFs and upload to Pinecone:

```bash
python store_index.py
```

This will:
- Load PDFs from `data/` directory
- Split text into chunks
- Generate embeddings using sentence transformers
- Upload to Pinecone index named "health-gpt"

## 🚀 Usage

### Starting the Application

```bash
python app.py
```

The application will start on `http://localhost:8080`

### Using the Chatbot

1. Open your browser and navigate to `http://localhost:8080`
2. Type your health-related question in the chat input
3. Press Enter or click the send button
4. The chatbot will retrieve relevant medical context and generate a response

### Example Queries

- "What are the symptoms of diabetes?"
- "How does high blood pressure affect the heart?"
- "What are the side effects of aspirin?"
- "Explain the difference between Type 1 and Type 2 diabetes"
- "What foods should I avoid with high cholesterol?"

## 📁 Project Structure

```
Health-GPT/
├── app.py                      # Main Flask application
├── store_index.py              # Script to index PDFs to Pinecone
├── template.py                 # Project structure generator
├── setup.py                    # Package setup
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in repo)
│
├── src/
│   ├── __init__.py
│   ├── helper.py              # Utility functions (PDF loading, embeddings)
│   └── prompt.py              # LLM prompt templates
│
├── model/
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin  # Llama 2 model (download separately)
│
├── data/                      # Medical PDF documents
│   └── *.pdf
│
├── static/
│   ├── css/
│   │   └── style.css         # Chat UI styling
│   └── images/
│       └── nurse.png         # Chatbot avatar
│
├── templates/
│   └── chat.html             # Chat interface template
│
└── research/
    └── trials.ipynb          # Experimentation notebook
```

## ⚙️ Configuration

### Model Settings (in app.py)

```python
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 512,      # Maximum response length
        'temperature': 0.8          # Creativity (0.0-1.0)
    }
)
```

### Retrieval Settings

```python
retriever = docsearch.as_retriever(
    search_kwargs={'k': 2}          # Number of documents to retrieve
)
```

### Pinecone Index

- **Index Name**: `health-gpt`
- **Dimension**: Automatically set by embedding model
- **Metric**: Cosine similarity

## 🔧 How It Works

### 1. Document Indexing Pipeline

```
PDF Files → Text Extraction → Chunking → Embedding Generation → Pinecone Upload
```

**store_index.py** performs:
1. Loads all PDFs from `data/` directory
2. Extracts text and splits into semantic chunks
3. Generates embeddings using HuggingFace sentence transformers
4. Uploads embeddings and text to Pinecone vector database

### 2. Query Pipeline (RAG)

```
User Query → Embedding → Similarity Search → Context Retrieval → LLM Generation → Response
```

**app.py** handles:
1. User sends a health question via chat interface
2. Question is embedded using same model as documents
3. Pinecone retrieves top-k most similar document chunks
4. Retrieved context + question fed to Llama 2
5. LLM generates contextually-grounded response
6. Response displayed in chat interface

### 3. Prompt Template

The system uses a custom prompt template (in `src/prompt.py`) that instructs the LLM to:
- Use only the provided context from retrieved documents
- Provide accurate medical information
- Admit when information is not in the context
- Be helpful and conversational

## 🎨 Chat Interface

The web interface features:
- **Real-time messaging** with timestamp display
- **User and bot avatars** for clear conversation flow
- **Responsive design** that works on mobile and desktop
- **Bootstrap styling** for clean, modern appearance
- **AJAX communication** for smooth, no-refresh experience

## 🧪 Development

### Running in Debug Mode

```bash
python app.py
# Debug mode is enabled by default in app.py
```

### Adding New Medical Documents

1. Place new PDFs in `data/` directory
2. Re-run the indexing script:
```bash
python store_index.py
```
3. Restart the application

### Customizing the Model

To use a different Llama 2 variant or another GGML model:
1. Download the desired model
2. Update the model path in `app.py`
3. Adjust `max_new_tokens` and `temperature` as needed

## 🚧 Limitations & Considerations

- **Response Time**: Local LLM inference can be slow (5-30 seconds per response)
- **Memory Usage**: Requires 8GB+ RAM for Llama 2 7B model
- **Medical Disclaimer**: This is an educational tool, not a substitute for professional medical advice
- **Context Window**: Limited to information in indexed documents
- **Language**: Primarily trained on English medical texts

## ⚠️ Medical Disclaimer

**IMPORTANT**: Health-GPT is designed for educational and informational purposes only. It should **NOT** be used as a substitute for:
- Professional medical advice
- Medical diagnosis
- Treatment recommendations
- Emergency medical services

**Always consult qualified healthcare professionals** for medical questions, symptoms, or treatment decisions. In case of medical emergency, call emergency services immediately.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 📚 Expand medical document corpus
- ⚡ Optimize inference speed
- 🎨 Enhance UI/UX
- 🔧 Add conversation history
- 📊 Implement source citation display
- 🌐 Multi-language support

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

## 🛣️ Future Enhancements

- [ ] Add conversation memory for context-aware follow-ups
- [ ] Display source documents with citations
- [ ] Implement user authentication
- [ ] Add export chat history feature
- [ ] Support for medical image analysis
- [ ] Multi-language medical knowledge base
- [ ] Integration with medical APIs (drug databases, clinical trials)
- [ ] Fine-tune model on medical QA datasets

## 📧 Contact

**Author**: Mohammed Azeezulla  
**Email**: mmoha134@depaul.edu  
**GitHub**: [@zeeza18](https://github.com/zeeza18)

## 🙏 Acknowledgments

- **Llama 2** by Meta AI - Base language model
- **TheBloke** - Quantized GGML models on HuggingFace
- **LangChain** - RAG framework and tools
- **Pinecone** - Vector database infrastructure
- **HuggingFace** - Sentence transformer embeddings

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for accessible health information**

[⭐ Star on GitHub](https://github.com/zeeza18/Health-GPT) • [🐛 Report Bug](https://github.com/zeeza18/Health-GPT/issues) • [💡 Request Feature](https://github.com/zeeza18/Health-GPT/issues)

</div>
