"""End-to-end API flow (ingest -> chat) with Qdrant, Redis and the LLM faked."""

import os
import re

QDRANT_DOC = (
    "Qdrant is a vector database written in Rust.\n"
    "It stores embeddings and supports filtering by payload fields.\n\n"
    "Redis keeps the chat history for each session."
)
COOKING_DOC = "Pasta should be cooked in salted boiling water. Carbonara uses eggs and cheese."


def ingest(client, name, content, strategy="paragraph", **form):
    data = {"chunking_strategy": strategy, **{k: str(v) for k, v in form.items()}}
    return client.post("/api/ingest", files={"file": (name, content.encode())}, data=data)


def test_ingest_then_chat_returns_answer_with_sources(app_fakes):
    client, llm = app_fakes["client"], app_fakes["llm"]

    response = ingest(client, "qdrant.txt", QDRANT_DOC, chunk_size=20)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_chunks"] >= 2
    assert (body["chunk_size"], body["chunk_overlap"]) == (20, 0)
    assert ingest(client, "cooking.txt", COOKING_DOC, strategy="fixed").status_code == 200

    response = client.post("/api/chat", json={"session_id": "s1", "message": "What language is Qdrant written in?"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["response"].startswith("ANSWER")
    assert "written in Rust" in body["response"]  # retrieved chunk reached the LLM
    assert body["sources"][0]["filename"] == "qdrant.txt"
    assert set(body["sources"][0]) == {"document_id", "filename", "chunk_index", "score", "snippet"}
    assert all("cooking" not in s["filename"] for s in body["sources"])  # below min score
    # First message: intent classified, no rewrite needed.
    assert len(llm.calls_of("You classify")) == 1
    assert llm.calls_of("standalone search query") == []


def test_follow_up_question_is_rewritten_before_retrieval(app_fakes):
    client, llm, embedding = app_fakes["client"], app_fakes["llm"], app_fakes["embedding"]
    llm.rewrite = lambda message: "Which language is Qdrant written in?"
    ingest(client, "qdrant.txt", QDRANT_DOC)

    client.post("/api/chat", json={"session_id": "s1", "message": "What is Qdrant?"})
    response = client.post("/api/chat", json={"session_id": "s1", "message": "What language is it written in?"})

    assert response.status_code == 200
    rewrite_call = llm.calls_of("standalone search query")[-1]
    assert "What is Qdrant?" in rewrite_call["user"]  # history was included
    assert embedding.queries[-1] == "Which language is Qdrant written in?"
    # The answer is still generated for the user's original message.
    answer_call = llm.calls[-1]
    assert answer_call["messages"][-1] == {"role": "user", "content": "What language is it written in?"}


def test_chat_filters_top_k_and_min_score(app_fakes):
    client = app_fakes["client"]
    qdrant_id = ingest(client, "qdrant.txt", QDRANT_DOC, chunk_size=20).json()["document_id"]
    cooking_id = ingest(client, "cooking.txt", COOKING_DOC, strategy="fixed").json()["document_id"]

    def sources(**extra):
        body = client.post("/api/chat", json={"session_id": "s", "message": "Qdrant Redis pasta", **extra}).json()
        return body["sources"]

    assert {s["filename"] for s in sources(document_ids=[cooking_id])} == {"cooking.txt"}
    assert {s["document_id"] for s in sources(chunking_strategy="paragraph")} == {qdrant_id}
    assert len(sources(top_k=1)) == 1
    assert sources(document_ids=[999]) == []

    response = client.post("/api/chat", json={"session_id": "s", "message": "hi", "top_k": 1000})
    assert response.status_code == 422


def test_question_about_interviews_goes_to_rag_not_booking(app_fakes):
    client, llm = app_fakes["client"], app_fakes["llm"]
    ingest(client, "hr.txt", "The interview schedule: interviews run Monday to Friday, 9am to 5pm.")

    response = client.post(
        "/api/chat", json={"session_id": "s1", "message": "What's the interview schedule in the doc?"}
    )

    body = response.json()
    assert body["sources"] and body["sources"][0]["filename"] == "hr.txt"
    assert llm.calls_of("Extract interview booking") == []
    assert client.get("/api/bookings").json() == []


def test_multi_turn_booking_over_api(app_fakes):
    client, llm = app_fakes["client"], app_fakes["llm"]
    llm.classify = lambda message, system: "booking"

    def extract(message):
        fields = {}
        if match := re.search(r"[\w.]+@[\w.]+", message):
            fields["email"] = match.group()
        if "Jane Doe" in message:
            fields["name"] = "Jane Doe"
        if match := re.search(r"\d{4}-\d{2}-\d{2}", message):
            fields["date"] = match.group()
        if match := re.search(r"\b\d{1,2}(:\d{2})?\s?[ap]m\b|\b\d{1,2}:\d{2}\b", message):
            fields["time"] = match.group()
        return fields

    llm.extract = extract

    def chat(message, session="s1"):
        response = client.post("/api/chat", json={"session_id": session, "message": message})
        assert response.status_code == 200, response.text
        return response.json()

    first = chat("I want to book an interview. I'm Jane Doe, jane@example.com")
    assert "the date" in first["response"] and "the time" in first["response"]
    assert first["sources"] == []

    second = chat("2099-10-20 at 2pm")
    assert "booked" in second["response"]

    bookings = client.get("/api/bookings").json()
    assert len(bookings) == 1
    assert bookings[0]["email"] == "jane@example.com"
    assert bookings[0]["time"] == "14:00:00"

    # Same slot from another session is refused.
    other = chat("Book Jane Doe other@example.com 2099-10-20 2pm", session="s2")
    assert "already booked" in other["response"]
    assert len(client.get("/api/bookings").json()) == 1


def test_documents_list_and_delete(app_fakes, test_settings):
    client, store = app_fakes["client"], app_fakes["vector_store"]
    doc_id = ingest(client, "qdrant.txt", QDRANT_DOC).json()["document_id"]
    assert len(os.listdir(test_settings.upload_dir)) == 1

    documents = client.get("/api/documents").json()
    assert [d["id"] for d in documents] == [doc_id]
    assert "raw_text" not in documents[0]

    response = client.delete(f"/api/documents/{doc_id}")
    assert response.status_code == 200
    assert store.points == {}
    assert os.listdir(test_settings.upload_dir) == []
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_ingest_validation_errors(app_fakes, test_settings):
    client = app_fakes["client"]

    response = ingest(client, "a.txt", "hello world.", strategy="fixed", chunk_size=999)
    assert response.status_code == 400
    assert "chunk_size" in response.json()["detail"]

    response = ingest(client, "a.txt", "hello world.", strategy="fixed", chunk_size=10, overlap=10)
    assert response.status_code == 400

    response = ingest(client, "a.csv", "a,b", strategy="fixed")
    assert response.status_code == 400

    test_settings.max_upload_size_mb = 0
    response = ingest(client, "big.txt", "x" * 100)
    assert response.status_code == 413
    assert os.listdir(test_settings.upload_dir) == []


def test_qdrant_down_returns_503_and_keeps_nothing(app_fakes, test_settings):
    client, store = app_fakes["client"], app_fakes["vector_store"]
    store.available = False

    response = ingest(client, "qdrant.txt", QDRANT_DOC)
    assert response.status_code == 503
    assert response.json() == {"detail": "Qdrant is unavailable. Please try again later."}
    assert client.get("/api/documents").json() == []
    assert os.listdir(test_settings.upload_dir) == []
