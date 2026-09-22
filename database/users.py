
#sqlite 
import sqlite3

# shared helper used to open and manage sqlite connections
from database.connection import open_database

# cleans usernames so storage and lookup use a consistent format
def normalise_username(username):
    # convert the value to text, remove spaces and use lowercase
    return str(username or "").strip().lower()


# cleans email addresses so storage and lookup use a consistent format
def normalise_email(email):
    # convert the value to text, remove spaces and use lowercase
    return str(email or "").strip().lower()


# converts a sqlite user row into a normal python dictionary
def row_to_user(row):
    # return none when no user row was found
    return dict(row) if row is not None else None

# creates a new registered user account
def create_user(username, email, password_hash, display_name=""):
    # clean the username before validation and storage
    clean_username = normalise_username(username)
    # clean the email before validation and storage
    clean_email = normalise_email(email)
    # clean the already hashed password value
    clean_password_hash = str(password_hash or "").strip()
    # store an empty display name as none
    clean_display_name = str(display_name or "").strip() or None
    # a username is required for account creation
    if not clean_username:
        raise ValueError("A username is required.")
    # an email address is required for account creation
    if not clean_email:
        raise ValueError("An email address is required.")

    # only a prepared password hash should be passed into this database layer
    if not clean_password_hash:
        raise ValueError("A password hash is required.")

    # try to create the account and handle duplicate account details
    try:
        # open a managed database connection
        with open_database() as connection:

            # insert the new user account into the users table
            cursor = connection.execute(
                """
                INSERT INTO users (
                    username,
                    email,
                    password_hash,
                    display_name
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    # save the cleaned username
                    clean_username,
                    # save the cleaned email
                    clean_email,
                    # save the hashed password
                    clean_password_hash,
                    # save the optional display name
                    clean_display_name,
                ),
            )

            # get the database id created for the new account
            user_id = cursor.lastrowid

            # retrieve the newly created user record
            row = connection.execute(
                """
                SELECT *
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

        # convert and return the saved database row
        return row_to_user(row)

    # catch database uniqueness errors such as repeated usernames or emails
    except sqlite3.IntegrityError as error:

        # convert the database error message to lowercase for simple checking
        error_text = str(error).lower()

        # return a clear message when the username is already taken
        if "username" in error_text:
            raise ValueError(
                "This username is already in use."
            ) from error

        # return a clear message when the email is already registered
        if "email" in error_text:
            raise ValueError(
                "An account with this email already exists."
            ) from error

        # reraise any other integrity error that was not handled above
        raise

# retrieves a user account using its username
def get_user_by_username(username):
    # clean the username before using it in a query
    clean_username = normalise_username(username)
    # return nothing when the userame is empty
    if not clean_username:
        return None
    # look for a matching username in the users table
    with open_database() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (clean_username,),
        ).fetchone()

    # convert the database row before returning it
    return row_to_user(row)


# checks whether a username has already been registered
def username_exists(username):
    #True when a username is already in use
    # a returned user record means the username already exists
    return get_user_by_username(username) is not None


# retrieves a user account using its email address
def get_user_by_email(email):
    #r eturn a user by email address

    # clean the email before using it in a query
    clean_email = normalise_email(email)

    # return nothing when the email is empty
    if not clean_email:
        return None

    # look for a matching email in the users table
    with open_database() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (clean_email,),
        ).fetchone()

    # convert the database row before returning it
    return row_to_user(row)
# retrieves a user account using its database id
def get_user_by_id(user_id):
    # return user based on ID

    # try to convert the supplied user id into an integer
    try:
        clean_user_id = int(user_id)

    # invalid user ids cannot be queried
    except (TypeError, ValueError):
        return None

    # look for the matching user record
    with open_database() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM users
            WHERE id = ?
            """,
            (clean_user_id,),
        ).fetchone()

    # convert the database row before returning it
    return row_to_user(row)


# checks whether an email address has already been registered
def email_exists(email):
    ## True when an email is already registere
    # a returned user record means the email already exists
    return get_user_by_email(email) is not None

# changes whether a user account is active or inactive
def set_user_active(user_id, is_active):
    # make sure the user id is a valid integer
    try:
        clean_user_id = int(user_id)

    # return false when the user id cannot be used
    except (TypeError, ValueError):
        return False

    # sqlite stores the active state as 1 or 0
    active_value = 1 if is_active else 0

    # update the active state for the selected account
    with open_database() as connection:
        cursor = connection.execute(
            """
            UPDATE users
            SET
                is_active = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (active_value, clean_user_id),
        )

        # return true only when a user row was actually updated
        return cursor.rowcount > 0


# replaces the stored password hash for a user account
def update_user_password(user_id, password_hash):
    ##update the stored password hash for a user
    # make sure the user id is valid
    try:
        clean_user_id = int(user_id)

    # return false when the user id cannot be converted
    except (TypeError, ValueError):
        return False

    # clean the supplied password hash before saving it
    clean_password_hash = str(password_hash or "").strip()

    # do not update the account with an empty password hash
    if not clean_password_hash:
        return False

    # update the stored password hash and modification timestamp
    with open_database() as connection:
        cursor = connection.execute(
            """
            UPDATE users
            SET
                password_hash = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (clean_password_hash, clean_user_id),
        )

        # return true only when a matching account was updated
        return cursor.rowcount > 0

# permanently removes a user account from the database
def delete_user(user_id):
    # make sure the user id is valid before running the delete query
    try:
        clean_user_id = int(user_id)

    # return false when the supplied id is invalid
    except (TypeError, ValueError):
        return False

    # delete the matching account from the users table
    with open_database() as connection:
        cursor = connection.execute(
            """
            DELETE FROM users
            WHERE id = ?
            """,
            (clean_user_id,),
        )
        # return true only when a user record was actually deleted
        return cursor.rowcount > 0