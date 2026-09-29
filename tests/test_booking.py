from datetime import date, datetime, time

import pytest

from app.db.models import Booking
from app.services.booking_service import BOOKING_STATE, BookingService
from app.services.booking_validation import parse_date, parse_time, validate_draft
from tests.conftest import FakeLLM, FakeMemoryService

NOW = datetime(2026, 9, 29, 10, 0)
FULL_MESSAGE = "Book an interview: Jane Doe, jane@example.com, 2026-10-20 at 2pm"
FULL_FIELDS = {"name": "Jane Doe", "email": "jane@example.com", "date": "2026-10-20", "time": "2pm"}


# --- Parsing and validation -------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("9:00", time(9, 0)),
        ("09:00", time(9, 0)),
        ("14:30", time(14, 30)),
        ("14:30:00", time(14, 30)),
        ("2pm", time(14, 0)),
        ("2 PM", time(14, 0)),
        ("2:30pm", time(14, 30)),
        ("11 a.m.", time(11, 0)),
        ("12am", time(0, 0)),
        ("12pm", time(12, 0)),
        ("noon", time(12, 0)),
    ],
)
def test_parse_time_accepts_common_formats(value, expected):
    assert parse_time(value) == expected


@pytest.mark.parametrize("value", ["9", "25:00", "13pm", "0am", "9:75", "half past two"])
def test_parse_time_rejects_invalid(value):
    with pytest.raises(ValueError):
        parse_time(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-10-05", date(2026, 10, 5)),
        ("2026/10/05", date(2026, 10, 5)),
        ("October 5th, 2026", date(2026, 10, 5)),
        ("5 Oct 2026", date(2026, 10, 5)),
    ],
)
def test_parse_date_accepts_common_formats(value, expected):
    assert parse_date(value) == expected


@pytest.mark.parametrize("value", ["2026-02-30", "tomorrow", "05/10/2026", ""])
def test_parse_date_rejects_invalid(value):
    with pytest.raises(ValueError):
        parse_date(value)


def test_validate_draft_complete_and_valid():
    result = validate_draft(FULL_FIELDS, now=NOW)
    assert result.details is not None
    assert result.details.name == "Jane Doe"
    assert result.details.date == date(2026, 10, 20)
    assert result.details.time == time(14, 0)


def test_validate_draft_reports_missing_fields():
    result = validate_draft({"name": "Jane Doe"}, now=NOW)
    assert result.details is None
    assert result.missing == ["email", "date", "time"]
    assert result.errors == {}


def test_validate_draft_reports_invalid_email():
    result = validate_draft({**FULL_FIELDS, "email": "jane-at-example"}, now=NOW)
    assert result.details is None
    assert "email" in result.errors


@pytest.mark.parametrize(
    "draft",
    [
        {**FULL_FIELDS, "date": "2026-09-01"},                    # past date
        {**FULL_FIELDS, "date": "2026-09-29", "time": "9:00"},    # earlier today
    ],
)
def test_validate_draft_rejects_past_datetime(draft):
    result = validate_draft(draft, now=NOW)
    assert result.details is None
    assert "future" in result.errors["date"]


def test_validate_draft_rejects_bad_name():
    result = validate_draft({**FULL_FIELDS, "name": "R2D2"}, now=NOW)
    assert "name" in result.errors


# --- BookingService with a mocked LLM ---------------------------------------

def make_service(classify="booking", extract=None):
    llm = FakeLLM(
        classify=classify if callable(classify) else (lambda message, system: classify),
        extract=extract or (lambda message: {}),
    )
    memory = FakeMemoryService()
    return BookingService(llm, memory), llm, memory


def test_question_is_not_handled_as_booking(db):
    service, llm, _ = make_service(classify="question")
    assert service.handle_message(db, "s1", "What's the interview schedule in the doc?", now=NOW) is None
    assert llm.calls_of("Extract interview booking") == []


def test_classifier_output_is_parsed_leniently():
    service, _, _ = make_service(classify="Booking.")
    assert service.classify_intent("book me in") == "booking"
    service, _, _ = make_service(classify="I am not sure")
    assert service.classify_intent("hmm") == "question"
    # "cancel" is only valid while a booking is pending.
    service, _, _ = make_service(classify="cancel")
    assert service.classify_intent("stop") == "question"
    assert service.classify_intent("stop", pending={"name": "Jane"}) == "cancel"


def test_complete_booking_in_one_message(db):
    service, llm, memory = make_service(extract=lambda message: FULL_FIELDS)
    reply = service.handle_message(db, "s1", FULL_MESSAGE, now=NOW)

    assert "booked" in reply
    booking = db.query(Booking).one()
    assert (booking.name, booking.email, booking.date, booking.time) == (
        "Jane Doe", "jane@example.com", date(2026, 10, 20), time(14, 0),
    )
    assert memory.get_state("s1", BOOKING_STATE) is None
    assert llm.calls_of("Extract interview booking")[0]["json"] is True


def test_collects_missing_fields_across_turns(db):
    replies = {
        "I'd like to book an interview, I'm Jane Doe": {"name": "Jane Doe"},
        "jane@example.com, on 2026-10-20": {"email": "jane@example.com", "date": "2026-10-20"},
        "at 9:30": {"time": "9:30"},
    }
    service, llm, memory = make_service(extract=lambda message: replies[message])
    messages = list(replies)

    first = service.handle_message(db, "s1", messages[0], now=NOW)
    assert "your email address" in first and "the date" in first and "the time" in first
    assert memory.get_state("s1", BOOKING_STATE) == {"name": "Jane Doe"}

    second = service.handle_message(db, "s1", messages[1], now=NOW)
    assert "the time" in second and "email" not in second

    # While a booking is pending, the classifier is told what is still missing.
    assert "the time" in llm.calls_of("You classify")[-1]["system"]

    third = service.handle_message(db, "s1", messages[2], now=NOW)
    assert "booked" in third
    assert db.query(Booking).one().time == time(9, 30)
    assert memory.get_state("s1", BOOKING_STATE) is None


def test_invalid_value_is_asked_for_again(db):
    service, _, memory = make_service(
        extract=lambda message: {**FULL_FIELDS, "email": "jane-at-example"}
        if "jane-at-example" in message else {"email": "jane@example.com"}
    )
    reply = service.handle_message(db, "s1", "Book Jane Doe jane-at-example 2026-10-20 2pm", now=NOW)
    assert "email" in reply and "isn't valid" in reply
    assert "email" not in memory.get_state("s1", BOOKING_STATE)

    reply = service.handle_message(db, "s1", "sorry, jane@example.com", now=NOW)
    assert "booked" in reply


def test_past_date_is_rejected(db):
    service, _, memory = make_service(
        extract=lambda message: {**FULL_FIELDS, "date": "2026-09-01"}
    )
    reply = service.handle_message(db, "s1", "Book Jane Doe jane@example.com 2026-09-01 2pm", now=NOW)
    assert "future" in reply
    assert db.query(Booking).count() == 0
    assert "date" not in memory.get_state("s1", BOOKING_STATE)


def test_duplicate_slot_is_rejected(db):
    service, _, _ = make_service(extract=lambda message: FULL_FIELDS)
    assert "booked" in service.handle_message(db, "s1", FULL_MESSAGE, now=NOW)

    other = {"name": "John Roe", "email": "john@example.com", "date": "2026-10-20", "time": "14:00"}
    service2, _, memory2 = make_service(extract=lambda message: other)
    reply = service2.handle_message(db, "s2", "Book John Roe john@example.com 2026-10-20 14:00", now=NOW)

    assert "already booked" in reply
    assert db.query(Booking).count() == 1
    # Name and email are kept; only the slot must be chosen again.
    assert memory2.get_state("s2", BOOKING_STATE) == {"name": "John Roe", "email": "john@example.com"}


def test_cancel_clears_pending_booking(db):
    service, _, memory = make_service(extract=lambda message: {"name": "Jane Doe"})
    service.handle_message(db, "s1", "Book an interview for Jane Doe", now=NOW)
    assert memory.get_state("s1", BOOKING_STATE) is not None

    service.llm_service.classify = lambda message, system: "cancel"
    reply = service.handle_message(db, "s1", "never mind, cancel", now=NOW)
    assert "cancelled" in reply
    assert memory.get_state("s1", BOOKING_STATE) is None


def test_extraction_ignores_values_not_in_message():
    service, _, _ = make_service(
        extract=lambda message: {"name": "Invented Person", "email": "made@up.com", "date": "2026-09-29", "time": "9:00"}
    )
    assert service.extract_fields("Book an interview please", today=NOW) == {}


def test_extraction_handles_invalid_json():
    service, _, _ = make_service(extract=lambda message: "not json at all")
    assert service.extract_fields("Book me for 2026-10-20", today=NOW) == {}


def test_extraction_drops_null_like_values():
    service, _, _ = make_service(
        extract=lambda message: {"name": "Jane Doe", "email": None, "date": "null", "time": ""}
    )
    assert service.extract_fields("I am Jane Doe", today=NOW) == {"name": "Jane Doe"}
