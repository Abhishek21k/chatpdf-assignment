from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone
from dotenv import load_dotenv
import os


class QueryProcessor:
    def __init__(self):
        load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "pdf-search"

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.openai_api_key,
        )

        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent assistant specializing in analyzing document content. 
            You have access to specific sections from documents that are most relevant to the user's question.
            
            Guidelines:
            1. Base your answers solely on the provided context
            2. Cite specific page numbers when referencing information
            3. If information spans multiple pages, mention all relevant pages
            4. If the context doesn't contain enough information, say so clearly
            5. If different pages have conflicting information, point this out
            6. Use direct quotes when appropriate, citing the page number
            7. Consider the relevance scores when weighing different pieces of information
            8. Mention the source document name when providing information
            
            Remember: You can only reference information that is explicitly present in the provided context."""),

            ("user", """Context Information:
            Document Sources: {sources}
            
            Relevant Passages:
            {context}
            
            Question: {question}
            
            Please provide a comprehensive answer based on the above context. Remember to cite pages and sources.""")
        ])

    def search_and_generate_response(self, query: str, top_k: int = 3):
        try:
            query_embedding = self.embeddings.embed_query(query)
            index = self.pc.Index(self.index_name)

            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )

            formatted_results = []
            context_text = ""
            sources_set = set()

            for match in results.matches:
                source = match.metadata['source']
                sources_set.add(source)

                result = {
                    'score': match.score,
                    'content': match.metadata['text'],
                    'page': match.metadata['page'],
                    'source': source,
                    'relevance': f"{match.score:.2%}"
                }
                formatted_results.append(result)

                context_text += f"\nSource: {source} (Page {result['page']}, Relevance: {
                    result['relevance']}):\n"
                context_text += f"{result['content']}\n"
                context_text += "-" * 80 + "\n"

            if context_text:
                sources_list = ", ".join(sources_set)

                # Generate response
                chain = self.prompt_template | self.llm
                llm_response = chain.invoke({
                    "sources": sources_list,
                    "context": context_text,
                    "question": query
                }).content
            else:
                llm_response = "I couldn't find any relevant information in the documents to answer your question."

            return {
                'llm_response': llm_response,
                'search_results': formatted_results,
                'sources': list(sources_set)
            }

        except Exception as e:
            print(f"Error during search and response generation: {str(e)}")
            return {
                'llm_response': "An error occurred while processing your question.",
                'search_results': [],
                'sources': []
            }
