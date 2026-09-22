
#  local application database
import sqlite3
# contextmanager allows the database connection to be opened and closed safely
from contextlib import contextmanager
# imports the configured database folder and database file path
from config.settings import DATABASE_DIR, DATABASE_PATH


# creates the database folder before a connection is opened
def ensure_database_directory():
    # parents are created as well if any part of the path does not exist
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)


# opens and manages a sqlite database connection
@contextmanager
def open_database():
    #open a SQsqlite Lite connection and handle commit or rollback.
    # make sure the database folder exists before opening the file
    ensure_database_directory()
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )
    # return database rows using column names instead of only tuple positions
    connection.row_factory = sqlite3.Row
    # enable sqlite foreign-key rules for this connection
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        # give the open connection to the code using this context manager
        yield connection
        # save changes when the database operation finishes successfully
        connection.commit()

    except Exception:
        # undo unfinished changes if an error occurs
        connection.rollback()
        raise
    finally:
        # always close the connection after the operation finishes
        connection.close()