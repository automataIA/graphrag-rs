"""
GraphRAG Python Bindings

Python bindings for the GraphRAG Rust library, providing high-performance
graph-based retrieval augmented generation.

Example:
    >>> import asyncio
    >>> from graphrag_py import PyGraphRAG
    >>>
    >>> async def main():
    >>>     rag = PyGraphRAG.default_local()
    >>>     await rag.add_document_from_text("Your document here")
    >>>     await rag.build_graph()
    >>>     answer = await rag.ask("Your question?")
    >>>     print(answer)
    >>>
    >>> asyncio.run(main())
"""

from graphrag_py import PyGraphRAG

__all__ = ["PyGraphRAG"]
