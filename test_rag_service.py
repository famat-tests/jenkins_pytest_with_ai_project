import pytest
from rag_service import index_documents, search_query, documents

@pytest.fixture
def setup_documents():
    # Set up the FAISS index with documents
    index_documents(documents)

def test_search_query(setup_documents):
    # Test case: Search query returns relevant documents
    query = "finance"
    results = search_query(query)
    assert len(results) > 0
    assert any("finance" in doc['doc_id'] for doc in results)

def test_empty_search_query(setup_documents):
    # Test case: Empty query should return no documents or low scores
    query = ""
    results = search_query(query)
    assert len(results) > 0
    assert all(result['score'] < 1 for result in results)

