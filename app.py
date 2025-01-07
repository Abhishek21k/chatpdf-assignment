import streamlit as st
from index import PDFProcessor
from query import QueryProcessor
import os
import tempfile


def clear_pinecone_index():
    try:
        processor = PDFProcessor()
        index = processor.pc.Index(processor.index_name)

        # Get all vectors
        vector_ids = []
        results = index.query(
            vector=[0.0] * 1536,
            top_k=10000,
            include_metadata=False
        )

        for match in results.matches:
            vector_ids.append(match.id)

        # Delete vectors in batches if any exist
        if vector_ids:
            index.delete(ids=vector_ids)

        st.session_state.processed_files = set()
        return True
    except Exception as e:
        st.error(f"Error clearing index: {str(e)}")
        return False


def main():
    st.set_page_config(page_title="PDF Chat App", layout="wide")
    st.title("PDF Chat Application")

    with st.sidebar:
        st.header("Settings")
        if st.button("Clear Memory"):
            if clear_pinecone_index():
                st.success("Successfully cleared all stored data!")
                st.session_state.processed_files = set()
            else:
                st.error("Failed to clear memory")

    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = QueryProcessor()
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.processed_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name

                with st.spinner('Processing PDF...'):
                    success = st.session_state.pdf_processor.process_pdf(
                        temp_path,
                        original_filename=uploaded_file.name
                    )
                    if success:
                        st.success(f"Successfully processed {
                                   uploaded_file.name}")
                        st.session_state.processed_files.add(
                            uploaded_file.name)
                    else:
                        st.error("Failed to process the PDF")

                os.unlink(temp_path)
            else:
                st.info("This PDF has already been processed")

        if st.session_state.processed_files:
            st.subheader("Processed Files:")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")

    with right_col:
        st.header("Ask Questions")

        if st.session_state.processed_files:
            query = st.text_input("Ask a question about your PDF(s):")
            top_k = st.slider("Number of results to display",
                              min_value=1, max_value=5, value=3)

            if query:
                with st.spinner('Searching and generating response...'):
                    response = st.session_state.query_processor.search_and_generate_response(
                        query, top_k=top_k)

                if response['llm_response']:
                    st.subheader("Answer:")
                    st.write(response['llm_response'])

                    if response['search_results']:
                        st.subheader("Source Passages:")
                        for i, result in enumerate(response['search_results'], 1):
                            pdf_name = os.path.basename(result['source'])
                            with st.expander(f"Source {i} (Score: {result['score']:.4f})"):
                                st.markdown(f"**Source PDF:** {pdf_name}")
                                st.markdown(f"**Page:** {result['page']}")
                                st.markdown("**Content:**")
                                st.write(result['content'])
                                st.divider()
                else:
                    st.warning("No relevant information found")
        else:
            st.info("Please upload and process a PDF first")


if __name__ == "__main__":
    main()
