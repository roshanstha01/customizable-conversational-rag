from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

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
    """Create tables and add columns introduced after the first release.

    create_all() never alters existing tables, so older app.db files would be
    missing new columns without this.
    """
    from app.db import models  # noqa: F401  (register models on Base)

    Base.metadata.create_all(bind=engine)

    existing = {column["name"] for column in inspect(engine).get_columns("documents")}
    if "stored_filename" not in existing:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE documents ADD COLUMN stored_filename VARCHAR(255)"))
