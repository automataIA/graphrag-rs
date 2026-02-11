"""
Document Q&A Example

This example demonstrates using GraphRAG for question-answering
over a collection of documents about software engineering topics.
"""

import asyncio
from graphrag_py import PyGraphRAG


# Sample documents about software engineering
DOCUMENTS = [
    """
    Test-Driven Development (TDD) is a software development practice where tests are written
    before the actual code. The TDD cycle consists of three steps: Red, Green, Refactor.
    First, write a failing test (Red). Then, write minimal code to make the test pass (Green).
    Finally, refactor the code to improve its design while keeping tests passing (Refactor).
    TDD helps ensure code correctness and encourages better design decisions.
    """,

    """
    Continuous Integration (CI) is a DevOps practice where developers frequently merge their
    code changes into a central repository. After each merge, automated builds and tests run
    to detect integration errors quickly. Popular CI tools include Jenkins, GitHub Actions,
    GitLab CI, and CircleCI. CI helps teams deliver software faster and with higher quality
    by catching bugs early in the development cycle.
    """,

    """
    Microservices architecture is a design approach where an application is structured as
    a collection of loosely coupled services. Each service is independently deployable,
    scalable, and maintainable. Services communicate through well-defined APIs, typically
    using REST or message queues. While microservices offer flexibility and scalability,
    they also introduce complexity in areas like service discovery, distributed tracing,
    and data consistency across services.
    """,

    """
    Domain-Driven Design (DDD) is an approach to software development that emphasizes
    collaboration between technical experts and domain experts. DDD introduces concepts
    like Bounded Contexts, Aggregates, Entities, and Value Objects to model complex
    business domains. The goal is to create software that reflects the real-world
    business processes and speaks the language of domain experts (Ubiquitous Language).
    """,
]


async def setup_knowledge_base():
    """Set up the GraphRAG knowledge base with documents."""
    print("🚀 Initializing GraphRAG...")
    rag = PyGraphRAG.default_local()

    print("📚 Loading documents...")
    for i, doc in enumerate(DOCUMENTS, 1):
        await rag.add_document_from_text(doc)
        print(f"   ✓ Document {i}/{len(DOCUMENTS)} loaded")

    print(f"\n📊 Status: {rag.has_documents()} documents loaded")

    print("\n🔨 Building knowledge graph...")
    print("   (This may take a moment with Ollama...)")

    try:
        await rag.build_graph()
        print("   ✓ Knowledge graph built successfully!")
    except Exception as e:
        print(f"   ❌ Error building graph: {e}")
        print("\n💡 Tip: Make sure Ollama is running:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Start the server: ollama serve")
        print("   3. Pull a model: ollama pull llama3")
        return None

    return rag


async def ask_questions(rag):
    """Ask various questions about the documents."""
    questions = [
        "What is Test-Driven Development?",
        "What are the three steps in the TDD cycle?",
        "What is Continuous Integration?",
        "Name some popular CI tools",
        "What is microservices architecture?",
        "What are the benefits and challenges of microservices?",
        "What is Domain-Driven Design?",
        "What concepts does DDD introduce?",
    ]

    print("\n" + "="*70)
    print("❓ Question & Answer Session")
    print("="*70)

    for i, question in enumerate(questions, 1):
        print(f"\n[Q{i}] {question}")
        try:
            answer = await rag.ask(question)
            print(f"[A{i}] {answer}")
        except Exception as e:
            print(f"[Error] {e}")


async def compare_with_reasoning(rag):
    """Use reasoning for complex comparative questions."""
    print("\n" + "="*70)
    print("🧠 Complex Question with Reasoning")
    print("="*70)

    complex_question = """
    Compare Test-Driven Development and Domain-Driven Design.
    How do they complement each other in software development?
    """

    print(f"\n[Q] {complex_question.strip()}")

    try:
        answer = await rag.ask_with_reasoning(complex_question)
        print(f"\n[A] {answer}")
    except Exception as e:
        print(f"[Error] {e}")


async def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("📖 Document Q&A with GraphRAG")
    print("="*70)
    print()

    # Setup
    rag = await setup_knowledge_base()
    if rag is None:
        return

    # Simple questions
    await ask_questions(rag)

    # Complex reasoning
    await compare_with_reasoning(rag)

    print("\n" + "="*70)
    print("✅ Example completed successfully!")
    print("="*70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
