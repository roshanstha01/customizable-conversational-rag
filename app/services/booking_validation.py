"""Validation for interview booking details collected from chat."""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, ValidationError, ValidationInfo, field_validator, model_validator

from app.config import local_now

BOOKING_FIELDS = ("name", "email", "date", "time")

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
)

TIME_PATTERN = re.compile(
    r"^(?P<hour>\d{1,2})"
    r"(?:[:.](?P<minute>\d{2}))?"
    r"(?::(?P<second>\d{2}))?"
    r"\s*(?P<meridiem>[ap])?\.?\s*(?:m\.?)?$",
    re.IGNORECASE,
)
ORDINAL_SUFFIX = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
NAME_PATTERN = re.compile(r"^[^\W\d_]+(?:[ .'\-][^\W\d_]+)*\.?$")


def parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = ORDINAL_SUFFIX.sub(r"\1", str(value).strip()).replace(",", " ")
    text = " ".join(text.split())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError("Use a date like 2026-10-05.")


def parse_time(value: Any) -> time:
    """Accept 24h times (9:00, 14:30) and 12h times (2pm, 2:30 PM, 11 a.m.)."""
    if isinstance(value, time):
        return value

    text = str(value).strip().lower()
    if text == "noon":
        return time(12, 0)

    match = TIME_PATTERN.match(text)
    if not match:
        raise ValueError("Use a time like 9:00, 14:30 or 2pm.")

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    meridiem = match.group("meridiem")

    if minute > 59:
        raise ValueError("Minutes must be between 00 and 59.")

    if meridiem:
        if not 1 <= hour <= 12:
            raise ValueError("With am/pm the hour must be between 1 and 12.")
        hour = hour % 12 + (12 if meridiem == "p" else 0)
    else:
        if match.group("minute") is None:
            # A bare "9" is ambiguous (9am or 9pm?).
            raise ValueError("Include minutes or am/pm, e.g. 9:00 or 9am.")
        if hour > 23:
            raise ValueError("Hour must be between 0 and 23.")

    return time(hour, minute)


class BookingDetails(BaseModel):
    """A complete, validated booking. Validate with context={"now": datetime} to
    control what counts as the future (defaults to the current time)."""

    name: str
    email: EmailStr
    date: date
    time: time

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not 1 <= len(value) <= 100 or not NAME_PATTERN.match(value):
            raise ValueError("Please give a name using letters only.")
        return value

    @field_validator("date", mode="before")
    @classmethod
    def validate_date(cls, value: Any) -> date:
        return parse_date(value)

    @field_validator("time", mode="before")
    @classmethod
    def validate_time(cls, value: Any) -> time:
        return parse_time(value)

    @model_validator(mode="after")
    def validate_in_future(self, info: ValidationInfo) -> "BookingDetails":
        now = (info.context or {}).get("now") or local_now()
        if datetime.combine(self.date, self.time) <= now:
            raise ValueError("The interview date and time must be in the future.")
        return self


@dataclass
class DraftValidation:
    details: Optional[BookingDetails] = None
    missing: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)


def validate_draft(draft: Dict[str, Optional[str]], now: Optional[datetime] = None) -> DraftValidation:
    """Validate a possibly incomplete booking.

    Returns the validated details when everything is present and valid;
    otherwise the missing fields and a message per invalid field.
    """
    missing = [name for name in BOOKING_FIELDS if not draft.get(name)]
    try:
        details = BookingDetails.model_validate(
            {name: draft.get(name) for name in BOOKING_FIELDS},
            context={"now": now or local_now()},
        )
        return DraftValidation(details=details)
    except ValidationError as error:
        errors: Dict[str, str] = {}
        for item in error.errors():
            loc = item["loc"][0] if item["loc"] else None
            if loc in missing:
                continue
            if loc is None:
                # Model-level check (date/time in the past).
                errors["date"] = str(item["ctx"]["error"]) if "ctx" in item else item["msg"]
            elif loc == "email":
                errors["email"] = "That doesn't look like a valid email address."
            else:
                ctx_error = item.get("ctx", {}).get("error")
                errors[loc] = str(ctx_error) if ctx_error else item["msg"]
        return DraftValidation(missing=missing, errors=errors)
