from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import uuid
import warnings
warnings.filterwarnings('ignore')


class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor with necessary configurations"""
        load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "pdf-search"

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.openai_api_key
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )

    def process_pdf(self, pdf_path: str, original_filename: str) -> bool:
        """
        Process a PDF file: load, split, embed, and store in Pinecone

        Args:
            pdf_path: Path to the PDF file
            original_filename: Original name of the uploaded file

        Returns:
            bool: Success status
        """
        try:
            print(f"\nLoading PDF from: {pdf_path}")

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages")

            # Split text into chunks
            print("\nSplitting text into chunks...")
            text_chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(text_chunks)} text chunks")

            # Get Pinecone index
            index = self.pc.Index(self.index_name)

            # Process chunks and upload to Pinecone
            print("\nStoring embeddings in Pinecone...")
            batch_size = 100
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]

                # Create embeddings for the batch
                texts = [doc.page_content for doc in batch]
                embeddings = self.embeddings.embed_documents(texts)

                # Prepare vectors for upload
                vectors = []
                for j, (text, embedding) in enumerate(zip(texts, embeddings)):
                    vector_id = str(uuid.uuid4())
                    metadata = {
                        "text": text,
                        "source": original_filename,  # Use original filename instead of temp path
                        "page": batch[j].metadata.get("page", 0)
                    }
                    vectors.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })

                # Upload batch to Pinecone
                index.upsert(vectors=vectors)
                print(f"Uploaded batch {
                      i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}")

            return True

        except Exception as e:
            print(f"\nError processing PDF: {str(e)}")
            return False


def main():
    """Main function to run the PDF processor"""
    print("Starting PDF processing...\n")

    try:
        processor = PDFProcessor()

        pdf_path = "./sqp302.pdf"

        success = processor.process_pdf(pdf_path)

        if success:
            print("\nPDF processing completed successfully!")
        else:
            print("\nPDF processing failed.")

    except Exception as e:
        print(f"\nError in main process: {str(e)}")


if __name__ == "__main__":
    main()
