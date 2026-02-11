#!/usr/bin/env python3
"""
Verification script for GraphRAG Python bindings installation.

This script checks that everything is installed and working correctly.
"""

import sys


def check_import():
    """Check if graphrag_py can be imported."""
    print("🔍 Checking import...")
    try:
        import graphrag_py
        print(f"   ✅ Successfully imported graphrag_py")
        print(f"   📦 Version: {graphrag_py.__version__}")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import: {e}")
        print("\n💡 Try running: uv run maturin develop")
        return False


def check_class_available():
    """Check if PyGraphRAG class is available."""
    print("\n🔍 Checking PyGraphRAG class...")
    try:
        from graphrag_py import PyGraphRAG
        print("   ✅ PyGraphRAG class is available")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import PyGraphRAG: {e}")
        return False


def check_instantiation():
    """Check if we can create an instance."""
    print("\n🔍 Checking instantiation...")
    try:
        from graphrag_py import PyGraphRAG
        rag = PyGraphRAG.default_local()
        print(f"   ✅ Successfully created instance: {rag}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create instance: {e}")
        return False


def check_methods():
    """Check if methods are available."""
    print("\n🔍 Checking available methods...")
    try:
        from graphrag_py import PyGraphRAG
        rag = PyGraphRAG.default_local()

        methods = [
            "add_document_from_text",
            "build_graph",
            "clear_graph",
            "ask",
            "ask_with_reasoning",
            "has_documents",
            "has_graph",
        ]

        for method in methods:
            if hasattr(rag, method):
                print(f"   ✅ {method}")
            else:
                print(f"   ❌ {method} not found")
                return False

        return True
    except Exception as e:
        print(f"   ❌ Error checking methods: {e}")
        return False


def check_async_support():
    """Check if async methods work."""
    print("\n🔍 Checking async support...")
    try:
        import asyncio
        from graphrag_py import PyGraphRAG

        async def test_async():
            rag = PyGraphRAG.default_local()
            # Just check that the method returns a coroutine
            result = rag.add_document_from_text("test")
            if hasattr(result, '__await__'):
                return True
            return False

        is_async = asyncio.run(test_async())
        if is_async:
            print("   ✅ Async methods are properly exposed")
            return True
        else:
            print("   ❌ Async methods not working correctly")
            return False
    except Exception as e:
        print(f"   ❌ Error checking async: {e}")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("GraphRAG Python Bindings - Installation Verification")
    print("="*60)

    checks = [
        ("Import", check_import),
        ("Class Availability", check_class_available),
        ("Instantiation", check_instantiation),
        ("Methods", check_methods),
        ("Async Support", check_async_support),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n📊 Results: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! GraphRAG Python bindings are ready to use.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\n💡 Common solutions:")
        print("   1. Run: uv run maturin develop")
        print("   2. Ensure Rust toolchain is installed")
        print("   3. Check that all dependencies are installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
