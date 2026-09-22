"""
SQLite database schema.

Creates the tables used for accounts, privacy preferences, check-ins,
saved reflections, resources and bookmarks.
"""

# imports the shared database connection helper used to create and update the schema
from database.connection import open_database
import json
from pathlib import Path
# path to the cleaned curated resources included with the project
RESOURCE_SEED_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "resources_seed.json"
)


# loads curated resources into a new database without replacing existing rows
def seed_resources(connection):

    # do nothing if the seed file is not available
    if not RESOURCE_SEED_PATH.exists():
        return

    # read the curated resource data
    with RESOURCE_SEED_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        resources = json.load(file)

    # add missing resources while preserving any rows already in the database
    connection.executemany(
        """
        INSERT OR IGNORE INTO resources (
            id,
            title,
            resource_type,
            source,
            topics_json,
            content,
            url,
            is_active,
            created_at,
            updated_at,
            daily_eligible
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                resource["id"],
                resource["title"],
                resource["resource_type"],
                resource["source"],
                resource["topics_json"],
                resource["content"],
                resource["url"],
                resource["is_active"],
                resource["created_at"],
                resource["updated_at"],
                resource["daily_eligible"],
            )
            for resource in resources
        ],
    )

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT,

    is_active INTEGER NOT NULL DEFAULT 1
        CHECK (is_active IN (0, 1)),

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


# creates the table that stores privacy choices for each user
# boolean privacy choices are stored as 0 or 1 in sqlite
# the privacy version records which version of the privacy notice was accepted
# deleting a user also removes the related privacy preferences
CREATE_PRIVACY_PREFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS privacy_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE,

    save_approved_checkins INTEGER NOT NULL DEFAULT 0
        CHECK (save_approved_checkins IN (0, 1)),

    use_saved_history_for_personalisation INTEGER NOT NULL DEFAULT 0
        CHECK (use_saved_history_for_personalisation IN (0, 1)),

    save_bookmarked_resources INTEGER NOT NULL DEFAULT 0
        CHECK (save_bookmarked_resources IN (0, 1)),

    privacy_version TEXT NOT NULL DEFAULT '1.0',
    accepted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);
"""


# creates the main table for completed check-ins that include structured questions
# core and context answers are stored as json text
# question, reflection and combined wellbeing scores are restricted to the 1 to 5 scale
# only approved or confirmed reflection results are stored with the check-in
# raw recordings, transcripts and raw model outputs are not stored in this table
# the user foreign key links each check-in to its account
CREATE_CHECKINS_TABLE = """
CREATE TABLE IF NOT EXISTS checkins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,

    question_set_version TEXT NOT NULL,
    checkin_status TEXT NOT NULL,
    reflection_mode TEXT,

    core_answers_json TEXT NOT NULL,
    context_answers_json TEXT NOT NULL,

    question_score REAL
        CHECK (
            question_score IS NULL
            OR question_score BETWEEN 1.0 AND 5.0
        ),

    reflection_score REAL
        CHECK (
            reflection_score IS NULL
            OR reflection_score BETWEEN 1.0 AND 5.0
        ),

    reflection_indicator TEXT,

    combined_wellbeing_score REAL
        CHECK (
            combined_wellbeing_score IS NULL
            OR combined_wellbeing_score BETWEEN 1.0 AND 5.0
        ),

    combined_wellbeing_indicator TEXT,

    approved_text_emotions_json TEXT NOT NULL DEFAULT '[]',
    approved_supportive_factors_json TEXT NOT NULL DEFAULT '[]',
    approved_strain_factors_json TEXT NOT NULL DEFAULT '[]',

    approved_vocal_observation TEXT,
    approved_visible_observation TEXT,

    confirmed_summary TEXT NOT NULL DEFAULT '',
    confirmed_appointment_points_json TEXT NOT NULL DEFAULT '[]',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);
"""


# creates an index on user_id so check-ins for one user can be found more efficiently
CREATE_CHECKINS_USER_INDEX = """
CREATE INDEX IF NOT EXISTS idx_checkins_user_id
ON checkins(user_id);
"""


# creates the table for reflection-only entries completed without structured questions
# the selected reflection mode and reflection score are stored here
# approved emotions, factors and observations are stored after user review
# the confirmed personal summary and appointment points are also stored
# structured question answers and question scores are deliberately not included
CREATE_SAVED_REFLECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS saved_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,

    reflection_mode TEXT NOT NULL DEFAULT '',

    reflection_score REAL
        CHECK (
            reflection_score IS NULL
            OR reflection_score BETWEEN 1.0 AND 5.0
        ),

    reflection_indicator TEXT,

    approved_text_emotions_json TEXT NOT NULL DEFAULT '[]',
    approved_supportive_factors_json TEXT NOT NULL DEFAULT '[]',
    approved_strain_factors_json TEXT NOT NULL DEFAULT '[]',

    approved_vocal_observation TEXT,
    approved_visible_observation TEXT,

    confirmed_summary TEXT NOT NULL DEFAULT '',
    confirmed_appointment_points_json TEXT NOT NULL DEFAULT '[]',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);
"""

# creates an index so reflection-only entries can be retrieved efficiently by user
CREATE_SAVED_REFLECTIONS_USER_INDEX = """
CREATE INDEX IF NOT EXISTS idx_saved_reflections_user_id
ON saved_reflections(user_id);
"""


# creates the table containing the curated postpartum resource collection
# topics are stored as json text because one resource can belong to several topics
# the active flag controls whether the resource is available in the application
# daily_eligible controls whether the resource may appear in the daily home-page rotation
CREATE_RESOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS resources (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    resource_type TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    topics_json TEXT NOT NULL DEFAULT '[]',
    content TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',

    is_active INTEGER NOT NULL DEFAULT 1
        CHECK (is_active IN (0, 1)),

    daily_eligible INTEGER NOT NULL DEFAULT 0
        CHECK (daily_eligible IN (0, 1)),

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


# creates the table that links user accounts with bookmarked resources
# the user and resource combination must be unique so the same resource is not bookmarked twice
# deleting either the user or resource also removes the related bookmark
CREATE_RESOURCE_BOOKMARKS_TABLE = """
CREATE TABLE IF NOT EXISTS resource_bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    resource_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (user_id, resource_id),

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    FOREIGN KEY (resource_id)
        REFERENCES resources(id)
        ON DELETE CASCADE
);
"""

# creates an index to make user bookmark lookups faster
CREATE_RESOURCE_BOOKMARKS_USER_INDEX = """
CREATE INDEX IF NOT EXISTS idx_resource_bookmarks_user_id
ON resource_bookmarks(user_id);
"""

# creates an index to make bookmark lookups by resource id faster
CREATE_RESOURCE_BOOKMARKS_RESOURCE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_resource_bookmarks_resource_id
ON resource_bookmarks(resource_id);
"""


# gets the names of all columns currently available in a database table
def get_table_columns(connection, table_name):
    # ask sqlite for information about the selected table structure
    rows = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    # return only the column names as a set for easy membership checking
    return {row["name"] for row in rows}


# adds newer score-related columns when an older check-ins table is being used
def ensure_checkin_score_columns(connection):
    # get the columns already present in the check-ins table
    columns = get_table_columns(connection, "checkins")

    # define newer columns that may be missing from an earlier development database
    missing_columns = {
        "reflection_mode": "TEXT",
        "question_score": "REAL",
        "reflection_score": "REAL",
        "reflection_indicator": "TEXT",
        "combined_wellbeing_score": "REAL",
        "combined_wellbeing_indicator": "TEXT",
    }

    # check each newer column and its sqlite data type
    for name, column_type in missing_columns.items():
        # only add the column when it is not already present
        if name not in columns:
            connection.execute(
                f"ALTER TABLE checkins ADD COLUMN {name} {column_type}"
            )


# adds reflection scoring columns when an older savedreflections table is used
def ensure_reflection_score_columns(connection):
    # read the current saved-reflections table structure
    columns = get_table_columns(connection, "saved_reflections")

    # define the reflection columns added during later development
    missing_columns = {
        "reflection_score": "REAL",
        "reflection_indicator": "TEXT",
    }

    # check whether each newer column already exists
    for name, column_type in missing_columns.items():
        # add only columns that are missing from the older table
        if name not in columns:
            connection.execute(
                f"ALTER TABLE saved_reflections ADD COLUMN {name} {column_type}"
            )


# adds the daily resource flag to resource tables created by older versions
def ensure_resource_daily_columns(connection):

    # get the columns currently stored in the resources table
    columns = get_table_columns(connection, "resources")

    # only update older databases that do not yet have this column
    if "daily_eligible" not in columns:
        connection.execute(
            """
            ALTER TABLE resources
            ADD COLUMN daily_eligible INTEGER NOT NULL DEFAULT 0
            """
        )


# creates the complete database schema and updates older development tables when needed
def initialise_database():

    # open one managed database connection for schema creation
    with open_database() as connection:

        # create the registered-user table
        connection.execute(CREATE_USERS_TABLE)
        # create the privacy-preferences table linked to users
        connection.execute(CREATE_PRIVACY_PREFERENCES_TABLE)
        # create the structured check-in table
        connection.execute(CREATE_CHECKINS_TABLE)
        # add newer scoring columns if this is an older development database
        ensure_checkin_score_columns(connection)
        # create the index used when retrieving check-ins by user
        connection.execute(CREATE_CHECKINS_USER_INDEX)
        # create the reflection-only table
        connection.execute(CREATE_SAVED_REFLECTIONS_TABLE)
        # add reflection score columns if an older table is detected
        ensure_reflection_score_columns(connection)
        # create the index used when retrieving saved reflections by user
        connection.execute(CREATE_SAVED_REFLECTIONS_USER_INDEX)
        # create the curated resources table
        connection.execute(CREATE_RESOURCES_TABLE)
        # add daily-resource support if an older resources table is detected
        ensure_resource_daily_columns(connection)
        # load the curated resource catalogue when rows are missing
        seed_resources(connection)
        # create the table linking users to bookmarked resources
        connection.execute(CREATE_RESOURCE_BOOKMARKS_TABLE)
        # create the index used when retrieving bookmarks for a user
        connection.execute(CREATE_RESOURCE_BOOKMARKS_USER_INDEX)
        # create the index used when looking up bookmarks for a resource
        connection.execute(CREATE_RESOURCE_BOOKMARKS_RESOURCE_INDEX)