from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os


class QueryProcessor:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "pdf-search"

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.openai_api_key,
            # dimensions=1536
        )

    def search(self, query: str, top_k: int = 3):
        """
        Search for similar content in the PDF

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of matching results with their content and metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)

            # Get Pinecone index
            index = self.pc.Index(self.index_name)

            # Query Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )

            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    'score': match.score,
                    'content': match.metadata['text'],
                    'page': match.metadata['page'],
                    'source': match.metadata['source']
                })

            return formatted_results

        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []


def main():
    try:
        # Initialize query processor
        processor = QueryProcessor()

        # Get user query
        query = input("\nEnter your search query: ")

        # Search
        print("\n🔍 Searching...")
        results = processor.search(query)

        # Display results
        if results:
            print("\n📝 Search Results:")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (Score: {result['score']:.4f})")
                print("-" * 50)
                print(f"Content: {result['content'][:200]}...")
                print(f"Page: {result['page']}")
                print(f"Source: {result['source']}")
        else:
            print("\n❌ No results found.")

    except Exception as e:
        print(f"Error during search: {str(e)}")


if __name__ == "__main__":
    print("🔎 PDF Search Query System")
    main()
