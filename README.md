# Chat with PDF Application

An interactive application that allows users to upload PDFs, process them, and ask questions about their content using natural language.

## Features

- PDF document upload and processing
- Text extraction and embedding generation
- Interactive question-answering interface
- Support for multiple PDF uploads
- Context-aware responses using vector similarity search

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Abhishek21k/chatpdf-assignment.git
cd chatpdf-assignment
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

4. Create a Pinecone index:

- Log in to your Pinecone account
- Create a new index named "pdf-search" with dimension 1536
- Choose the appropriate region and metric (cosine)

## Project Structure

```
.
├── app.py          # Streamlit interface
├── index.py        # PDF processing and embedding logic
├── query.py        # Search and retrieval functionality
├── requirements.txt
└── .env
```

## Running the Application

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Access the application in your web browser at `http://localhost:8501`

## Usage

1. **Upload PDF**

   - Click the "Choose a PDF file" button in the left column
   - Select your PDF file
   - Wait for the processing to complete

2. **Ask Questions**

   - Enter your question in the text input field
   - Adjust the number of results to display using the slider
   - View the relevant passages from the PDF with their source information

3. **Clear Memory**
   - Use the "Clear Memory" button in the sidebar to remove all processed PDFs
   - This will reset the system for new documents

## Technical Implementation

- Uses LangChain for PDF processing and text splitting
- OpenAI embeddings for vector representation
- Pinecone for vector storage and similarity search
- Streamlit for the user interface

## Limitations

- PDF size may affect processing time
- API usage costs apply for OpenAI embeddings
- Processing very large PDFs may require additional memory
