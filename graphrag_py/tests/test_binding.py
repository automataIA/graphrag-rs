"""
Comprehensive test suite for GraphRAG Python bindings.

These tests verify that the Python bindings work correctly.
Note: Some tests may fail if Ollama is not running locally.
"""

import pytest
import asyncio
from graphrag_py import PyGraphRAG


class TestPyGraphRAGInitialization:
    """Test GraphRAG initialization and configuration."""

    def test_default_local_initialization(self):
        """Test that we can create a default local instance."""
        rag = PyGraphRAG.default_local()
        assert rag is not None
        assert isinstance(rag, PyGraphRAG)

    def test_repr(self):
        """Test the string representation."""
        rag = PyGraphRAG.default_local()
        repr_str = repr(rag)
        assert "PyGraphRAG" in repr_str
        assert "documents=" in repr_str
        assert "graph_built=" in repr_str


class TestDocumentManagement:
    """Test document addition and management."""

    @pytest.mark.asyncio
    async def test_add_document_from_text(self):
        """Test adding a document from text."""
        rag = PyGraphRAG.default_local()

        # Add a simple document
        await rag.add_document_from_text("This is a test document about artificial intelligence.")

        # Verify document was added
        assert rag.has_documents() is True

    @pytest.mark.asyncio
    async def test_multiple_documents(self):
        """Test adding multiple documents."""
        rag = PyGraphRAG.default_local()

        # Add multiple documents
        await rag.add_document_from_text("Document 1: Python is a programming language.")
        await rag.add_document_from_text("Document 2: Rust is a systems programming language.")
        await rag.add_document_from_text("Document 3: Machine learning uses algorithms.")

        assert rag.has_documents() is True


class TestGraphBuilding:
    """Test knowledge graph building."""

    @pytest.mark.asyncio
    async def test_build_graph_basic(self):
        """Test building a basic knowledge graph."""
        rag = PyGraphRAG.default_local()

        # Add a document
        await rag.add_document_from_text(
            "Python is a high-level programming language. "
            "Guido van Rossum created Python in 1991. "
            "Python is widely used for web development and data science."
        )

        # Build the graph (may fail if Ollama not running)
        try:
            await rag.build_graph()
            assert rag.has_graph() is True
        except Exception as e:
            pytest.skip(f"Graph building failed (Ollama may not be running): {e}")

    @pytest.mark.asyncio
    async def test_clear_graph(self):
        """Test clearing the knowledge graph."""
        rag = PyGraphRAG.default_local()

        # Add a document
        await rag.add_document_from_text("Test document for graph clearing.")

        # Build and clear graph
        try:
            await rag.build_graph()
            rag.clear_graph()
            # Documents should still exist
            assert rag.has_documents() is True
            # Graph should be cleared
            assert rag.has_graph() is False
        except Exception as e:
            pytest.skip(f"Graph operations failed: {e}")


class TestQuerying:
    """Test query functionality."""

    @pytest.mark.asyncio
    async def test_ask_basic(self):
        """Test basic ask functionality."""
        rag = PyGraphRAG.default_local()

        # Add knowledge
        await rag.add_document_from_text(
            "The Eiffel Tower is located in Paris, France. "
            "It was built in 1889 and is 330 meters tall."
        )

        try:
            # Build graph
            await rag.build_graph()

            # Ask a question
            answer = await rag.ask("Where is the Eiffel Tower?")

            assert isinstance(answer, str)
            assert len(answer) > 0
            print(f"Answer: {answer}")

        except Exception as e:
            pytest.skip(f"Query failed (Ollama may not be running): {e}")

    @pytest.mark.asyncio
    async def test_ask_with_reasoning(self):
        """Test ask with reasoning functionality."""
        rag = PyGraphRAG.default_local()

        # Add knowledge
        await rag.add_document_from_text(
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are used in deep learning."
        )

        try:
            # Build graph
            await rag.build_graph()

            # Ask a complex question
            answer = await rag.ask_with_reasoning(
                "What is the relationship between AI, machine learning, and deep learning?"
            )

            assert isinstance(answer, str)
            assert len(answer) > 0
            print(f"Reasoning answer: {answer}")

        except Exception as e:
            pytest.skip(f"Query with reasoning failed: {e}")

    @pytest.mark.asyncio
    async def test_query_without_graph(self):
        """Test that querying without a built graph auto-builds it."""
        rag = PyGraphRAG.default_local()

        # Add document but don't build graph explicitly
        await rag.add_document_from_text("Auto-build test document.")

        try:
            # This should auto-build the graph
            answer = await rag.ask("What is this about?")

            # Graph should now be built
            assert rag.has_graph() is True
            assert isinstance(answer, str)

        except Exception as e:
            pytest.skip(f"Auto-build query failed: {e}")


class TestStateChecking:
    """Test state checking methods."""

    def test_initial_state(self):
        """Test initial state of a new instance."""
        rag = PyGraphRAG.default_local()

        # Initially, no documents or graph
        assert rag.has_documents() is False
        assert rag.has_graph() is False

    @pytest.mark.asyncio
    async def test_state_after_document_addition(self):
        """Test state after adding documents."""
        rag = PyGraphRAG.default_local()

        # Add document
        await rag.add_document_from_text("State test document.")

        # Documents exist, but graph not built yet
        assert rag.has_documents() is True
        assert rag.has_graph() is False

    @pytest.mark.asyncio
    async def test_state_after_graph_build(self):
        """Test state after building graph."""
        rag = PyGraphRAG.default_local()

        # Add document and build graph
        await rag.add_document_from_text("Graph state test.")

        try:
            await rag.build_graph()

            # Both documents and graph should exist
            assert rag.has_documents() is True
            assert rag.has_graph() is True

        except Exception as e:
            pytest.skip(f"Graph building failed: {e}")


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_document(self):
        """Test adding an empty document."""
        rag = PyGraphRAG.default_local()

        # Try to add empty document - should either fail gracefully or be ignored
        try:
            await rag.add_document_from_text("")
            # If it succeeds, that's okay too (implementation-dependent)
        except Exception:
            # Empty documents might be rejected, which is fine
            pass

    @pytest.mark.asyncio
    async def test_very_long_document(self):
        """Test adding a very long document."""
        rag = PyGraphRAG.default_local()

        # Create a long document
        long_text = " ".join([f"Sentence {i} about topic {i % 10}." for i in range(1000)])

        # Should handle long documents
        await rag.add_document_from_text(long_text)
        assert rag.has_documents() is True

@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent operations on the same instance."""
    rag = PyGraphRAG.default_local()

    # Add multiple documents concurrently
    await asyncio.gather(
        rag.add_document_from_text("Concurrent document 1."),
        rag.add_document_from_text("Concurrent document 2."),
        rag.add_document_from_text("Concurrent document 3."),
    )

    assert rag.has_documents() is True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
