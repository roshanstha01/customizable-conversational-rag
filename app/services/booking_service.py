import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import local_now
from app.db.models import Booking
from app.services.booking_validation import BOOKING_FIELDS, BookingDetails, validate_draft
from app.services.llm_service import OllamaService
from app.services.memory_service import RedisMemoryService

logger = logging.getLogger(__name__)

BOOKING_STATE = "booking"

INTENT_BOOKING = "booking"
INTENT_QUESTION = "question"
INTENT_CANCEL = "cancel"

CLASSIFY_PROMPT = (
    "You classify messages sent to an assistant that answers questions about uploaded "
    "documents and can also book interviews.\n"
    "Answer with exactly one word:\n"
    "- booking: the user wants to book or schedule an interview for themselves, or is "
    "giving their details (name, email, date, time) for a booking.\n"
    "- question: anything else, including questions ABOUT interviews, schedules or "
    "bookings mentioned in the documents.\n"
    "Examples:\n"
    "'I'd like to book an interview for next Monday' -> booking\n"
    "'Schedule me an interview, I'm Jane, jane@x.com' -> booking\n"
    "'What's the interview schedule in the doc?' -> question\n"
    "'How are interviews booked according to the policy?' -> question"
)

CLASSIFY_PENDING_PROMPT = (
    "\n\nThe user is in the middle of booking an interview and was asked for missing "
    "details ({missing}).\n"
    "- booking: the message provides or corrects booking details (for example just an "
    "email address, a date, a time or a name).\n"
    "- cancel: the user wants to stop or cancel the booking.\n"
    "- question: the user asks something unrelated instead.\n"
    "Answer with exactly one word: booking, cancel or question."
)

EXTRACT_PROMPT = (
    "Extract interview booking details from the user's message.\n"
    "Today is {today} ({weekday}).\n"
    'Return a JSON object with exactly these keys: "name", "email", "date", "time".\n'
    "- name: the person's name, only if they state it.\n"
    "- email: their email address.\n"
    "- date: the interview date as YYYY-MM-DD; resolve relative dates such as "
    "'tomorrow' or 'next Monday' using today's date. If no day or date is mentioned, "
    "date must be null.\n"
    "- time: the time exactly as the user wrote it (e.g. '9:00', '14:30', '2pm').\n"
    "Use null for anything not stated in the message. Never guess or invent values."
)

FIELD_PROMPTS = {
    "name": "your full name",
    "email": "your email address",
    "date": "the date (e.g. 2026-10-05)",
    "time": "the time (e.g. 9:00 or 2pm)",
}


# Messages without any of these words skip the LLM classifier (unless a booking is pending).
BOOKING_HINT = re.compile(r"\b(book|schedul|interview|appointment|meeting)", re.IGNORECASE)

DATE_HINT = re.compile(
    r"\d|\b(today|tomorrow|week|month|"
    r"(mon|tues?|wed(nes)?|thu(rs?)?|fri|sat(ur)?|sun)(day)?|"
    r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|"
    r"oct(ober)?|nov(ember)?|dec(ember)?)\b",
    re.IGNORECASE,
)
TIME_HINT = re.compile(r"\d|\b(noon)\b", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\S+@\S+")


def _supported_by_message(field: str, value: str, message: str) -> bool:
    """Guard against the LLM inventing values the user never gave."""
    text = message.lower()
    if field in ("name", "email"):
        return value.lower() in text
    # Digits or month-like words inside an email address are not a date/time.
    without_emails = EMAIL_PATTERN.sub(" ", message)
    if field == "date":
        return bool(DATE_HINT.search(without_emails))
    if field == "time":
        return bool(TIME_HINT.search(without_emails))
    return True


def _join(items: List[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class BookingService:
    """Handles interview booking conversations, keeping a partial booking per
    session in Redis until all details are collected."""

    def __init__(
        self,
        llm_service: OllamaService,
        memory_service: RedisMemoryService,
        state_ttl_seconds: int = 1800,
        timezone: Optional[str] = None,
    ) -> None:
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.state_ttl_seconds = state_ttl_seconds
        self.timezone = timezone

    def now(self) -> datetime:
        return local_now(self.timezone)

    def classify_intent(self, message: str, pending: Optional[Dict[str, Optional[str]]] = None) -> str:
        system_prompt = CLASSIFY_PROMPT
        allowed = {INTENT_BOOKING, INTENT_QUESTION}
        if pending is not None:
            missing = [FIELD_PROMPTS[name] for name in BOOKING_FIELDS if not pending.get(name)]
            system_prompt += CLASSIFY_PENDING_PROMPT.format(missing=_join(missing) or "none")
            allowed.add(INTENT_CANCEL)

        reply = self.llm_service.generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0,
        )

        words = reply.strip().lower().replace("'", " ").replace('"', " ").split()
        for word in words[:3]:
            word = word.strip(".,:;!-*")
            if word in allowed:
                return word
        logger.warning("Unrecognised intent label %r; treating as a question", reply)
        return INTENT_QUESTION

    def extract_fields(self, message: str, today: Optional[datetime] = None) -> Dict[str, Optional[str]]:
        today = today or self.now()
        reply = self.llm_service.generate_response(
            [
                {
                    "role": "system",
                    "content": EXTRACT_PROMPT.format(
                        today=today.date().isoformat(), weekday=today.strftime("%A")
                    ),
                },
                {"role": "user", "content": message},
            ],
            temperature=0,
            json_format=True,
        )

        try:
            data = json.loads(reply)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON for booking extraction: %r", reply)
            return {}
        if not isinstance(data, dict):
            return {}

        fields = {}
        for name in BOOKING_FIELDS:
            value = data.get(name)
            if value is None:
                continue
            value = str(value).strip()
            if not value or value.lower() in {"null", "none", "n/a", "unknown"}:
                continue
            if not _supported_by_message(name, value, message):
                logger.info("Ignoring extracted %s=%r not found in the message", name, value)
                continue
            fields[name] = value
        return fields

    def handle_message(
        self,
        db: Session,
        session_id: str,
        message: str,
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        """Return a reply if the message is part of a booking, else None (use RAG)."""
        pending = self.memory_service.get_state(session_id, BOOKING_STATE)
        if pending is None and not BOOKING_HINT.search(message):
            return None  # clearly not about booking; skip the LLM call

        now = now or self.now()
        intent = self.classify_intent(message, pending)
        logger.info("Session %s intent: %s (pending booking: %s)", session_id, intent, pending is not None)

        if intent == INTENT_CANCEL:
            self.memory_service.clear_state(session_id, BOOKING_STATE)
            return "No problem, I've cancelled that booking request."
        if intent != INTENT_BOOKING:
            return None

        draft = dict(pending or {})
        draft.update(self.extract_fields(message, today=now))

        result = validate_draft(draft, now=now)
        if result.details is None:
            for name in result.errors:
                draft.pop(name, None)  # ask for invalid values again
            self._save_draft(session_id, draft)
            return self._ask_for_details(result.missing, result.errors)

        reply = self._create_booking(db, result.details)
        if reply is None:
            self.memory_service.clear_state(session_id, BOOKING_STATE)
            details = result.details
            return (
                f"Your interview is booked for {details.date:%A, %B %d, %Y} at "
                f"{details.time:%H:%M} under {details.name} ({details.email})."
            )

        # Slot taken: keep name/email, ask for a different date/time.
        draft.pop("date", None)
        draft.pop("time", None)
        self._save_draft(session_id, draft)
        return reply

    def _save_draft(self, session_id: str, draft: Dict[str, Optional[str]]) -> None:
        self.memory_service.set_state(session_id, BOOKING_STATE, draft, self.state_ttl_seconds)

    def _create_booking(self, db: Session, details: BookingDetails) -> Optional[str]:
        """Save the booking; return an error reply if the slot is already taken."""
        slot_taken_reply = (
            f"Sorry, {details.date:%B %d, %Y} at {details.time:%H:%M} is already booked. "
            "Please choose a different date or time."
        )

        if db.query(Booking).filter_by(date=details.date, time=details.time).first():
            return slot_taken_reply

        booking = Booking(
            name=details.name,
            email=details.email,
            date=details.date,
            time=details.time,
        )
        db.add(booking)
        try:
            db.commit()
        except IntegrityError:
            # Another request took the slot between the check and the insert.
            db.rollback()
            return slot_taken_reply

        logger.info("Created booking %s for %s at %s %s", booking.id, details.email, details.date, details.time)
        return None

    @staticmethod
    def _ask_for_details(missing: List[str], errors: Dict[str, str]) -> str:
        parts = []
        for name, error in errors.items():
            parts.append(f"The {name} you gave isn't valid: {error}")
        if missing:
            parts.append(
                "To book your interview I still need "
                + _join([FIELD_PROMPTS[name] for name in missing])
                + "."
            )
        elif errors:
            parts.append("Please send the corrected details.")
        return " ".join(parts)
