import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 50)
print("   RAG Pipeline Test Suite")
print("=" * 50)

# ── Test 1: Config ─────────────────────────────────
def test_config():
    from src.utils.config import Config
    config = Config()
    assert config.EMBEDDING_MODEL is not None
    assert config.CHROMA_DIR is not None
    assert config.TOP_K > 0
    print("✅ Test 1 PASSED: Config loads correctly")

# ── Test 2: Document Loader ────────────────────────
# ✅ Correct - use the right function for .txt files
def test_document_loader():
    from src.utils.document_loader import load_and_chunk_pdf
    chunks = load_and_chunk_pdf("data/supportmanual.txt")
    assert len(chunks) > 0
    assert hasattr(chunks[0], "page_content")
    assert len(chunks[0].page_content) > 0
    print(f"✅ Test 2 PASSED: Document loader created {len(chunks)} chunks")

# ── Test 3: Graph State Structure ─────────────────
def test_graph_state():
    from src.agents.rag_graph import SupportState
    keys = SupportState.__annotations__.keys()
    assert "query" in keys
    assert "answer" in keys
    assert "escalated" in keys
    assert "retrieved_docs" in keys
    print("✅ Test 3 PASSED: GraphState has correct fields")

# ── Test 4: Routing Logic ──────────────────────────
def test_routing():
    from src.agents.rag_graph import route_intent

    # Escalated state → should go to hitl
    state1 = {"escalated": True, "escalation_reason": "test"}
    assert route_intent(state1) == "hitl"
    print("✅ Test 4a PASSED: Escalated queries route to HITL")

    # Normal state → should go to retriever
    state2 = {"escalated": False, "escalation_reason": ""}
    assert route_intent(state2) == "retriever"
    print("✅ Test 4b PASSED: Normal queries route to retriever")

# ── Test 5: HITL Manager ───────────────────────────
def test_hitl():
    from src.utils.hitl import HITLManager

    hitl = HITLManager(db_path="./test_hitl_temp.db")

    # Create ticket
    ticket_id = hitl.create_ticket(
        query="I want a refund urgently",
        intent="complaint",
        confidence=0.3
    )
    assert ticket_id.startswith("T-")
    print(f"✅ Test 5a PASSED: Ticket created → {ticket_id}")

    # Fetch ticket
    ticket = hitl.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket["status"] == "pending"
    print("✅ Test 5b PASSED: Ticket fetched successfully")

    # Resolve ticket
    success = hitl.resolve_ticket(ticket_id, "We will process your refund.")
    assert success is True
    print("✅ Test 5c PASSED: Ticket resolved successfully")

    # List tickets
    tickets = hitl.list_tickets()
    assert len(tickets) > 0
    print(f"✅ Test 5d PASSED: Listed {len(tickets)} ticket(s)")

    # Cleanup
    import os
    if os.path.exists("./test_hitl_temp.db"):
        os.remove("./test_hitl_temp.db")

# ── Test 6: Graph Build ────────────────────────────
def test_graph_builds():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from src.utils.config import Config
    config = Config()
    from src.agents.rag_graph import build_graph

    config_dict = {
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "top_k": config.TOP_K,
        "escalation_keywords": config.ESCALATION_KEYWORDS,
        "collection_name": config.COLLECTION_NAME,
        "chroma_dir": config.CHROMA_DIR,
        "groq_api_key": os.getenv("GROQ_API_KEY", "")
    }
    graph = build_graph(config_dict)
    assert graph is not None
    print("✅ Test 6 PASSED: LangGraph builds successfully")

# ── Run All Tests ──────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_config,
        test_document_loader,
        test_graph_state,
        test_routing,
        test_hitl,
        test_graph_builds,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__} → {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"   Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Project is complete.\n")
    else:
        print(f"\n⚠️  {failed} test(s) need fixing.\n")