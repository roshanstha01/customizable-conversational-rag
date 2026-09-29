import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables and add columns/indexes introduced after the first release.

    create_all() never alters existing tables, so older app.db files would be
    missing them without this.
    """
    from app.db import models  # noqa: F401  (register models on Base)

    Base.metadata.create_all(bind=engine)

    existing = {column["name"] for column in inspect(engine).get_columns("documents")}
    if "stored_filename" not in existing:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE documents ADD COLUMN stored_filename VARCHAR(255)"))

    try:
        with engine.begin() as connection:
            connection.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS uq_bookings_slot ON bookings (date, time)")
            )
    except (IntegrityError, OperationalError):
        # Existing duplicate bookings; the app still checks for taken slots before inserting.
        logger.warning("Could not add unique index on bookings(date, time): duplicate slots exist")
