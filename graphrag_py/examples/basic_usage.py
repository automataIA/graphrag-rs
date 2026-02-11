"""
Basic usage example for GraphRAG Python bindings.

This example demonstrates:
1. Creating a GraphRAG instance
2. Adding documents
3. Building the knowledge graph
4. Querying the system
"""

import asyncio
from graphrag_py import PyGraphRAG


async def main():
    print("="*60)
    print("GraphRAG Python Bindings - Basic Usage Example")
    print("="*60)

    # Step 1: Create a GraphRAG instance
    print("\n1. Creating GraphRAG instance...")
    rag = PyGraphRAG.default_local()
    print(f"   Created: {rag}")

    # Step 2: Add sample documents
    print("\n2. Adding documents...")

    doc1 = """
    Python is a high-level, interpreted programming language.
    It was created by Guido van Rossum and first released in 1991.
    Python emphasizes code readability with significant whitespace.
    It is widely used for web development, data science, artificial intelligence,
    and automation.
    """

    doc2 = """
    Rust is a multi-paradigm systems programming language.
    It was originally designed by Graydon Hoare at Mozilla Research.
    Rust was first released in 2010 and reached stability in 2015.
    Rust focuses on safety, especially safe concurrency, and performance.
    It is used for building operating systems, game engines, and web servers.
    """

    doc3 = """
    Machine learning is a subset of artificial intelligence.
    It enables systems to learn and improve from experience without being explicitly programmed.
    Machine learning algorithms build models based on sample data, known as training data.
    Common applications include image recognition, natural language processing, and recommendation systems.
    """

    await rag.add_document_from_text(doc1)
    print("   ✓ Added document about Python")

    await rag.add_document_from_text(doc2)
    print("   ✓ Added document about Rust")

    await rag.add_document_from_text(doc3)
    print("   ✓ Added document about Machine Learning")

    # Verify documents were added
    print(f"\n   Documents loaded: {rag.has_documents()}")

    # Step 3: Build the knowledge graph
    print("\n3. Building knowledge graph...")
    print("   (This may take a moment as it extracts entities and relationships)")

    try:
        await rag.build_graph()
        print("   ✓ Knowledge graph built successfully!")
        print(f"   Graph status: {rag.has_graph()}")
    except Exception as e:
        print(f"   ⚠ Graph building failed: {e}")
        print("   Note: Make sure Ollama is running (ollama serve)")
        return

    # Step 4: Query the system
    print("\n4. Querying the system...")

    questions = [
        "Who created Python?",
        "What is Rust used for?",
        "When was Rust first released?",
        "What is machine learning?",
    ]

    for question in questions:
        try:
            print(f"\n   Q: {question}")
            answer = await rag.ask(question)
            print(f"   A: {answer}")
        except Exception as e:
            print(f"   Error: {e}")

    # Step 5: Try reasoning-based query
    print("\n5. Testing complex query with reasoning...")

    complex_question = "Compare Python and Rust in terms of their design goals and use cases"

    try:
        print(f"\n   Q: {complex_question}")
        answer = await rag.ask_with_reasoning(complex_question)
        print(f"   A: {answer}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
