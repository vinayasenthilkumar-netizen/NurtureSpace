# regex is used for simple username and email format checks
import re
import sqlite3
# database helpers for creating users and checking existing accounts
from database.users import (
    create_user,
    email_exists,
    username_exists,
    get_user_by_username,
    normalise_email,
    normalise_username,
)


# username length limits used during registration
MINIMUM_USERNAME_LENGTH = 3
MAXIMUM_USERNAME_LENGTH = 30
# usernames can contain only letters, numbers and underscores
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
# password hashing and verification are kept in a separate service
from services.password_service import (
    hash_password,
    verify_password,
)
# password and display-name limits
MINIMUM_PASSWORD_LENGTH = 8
MAXIMUM_PASSWORD_LENGTH = 128
MAXIMUM_DISPLAY_NAME_LENGTH = 50
# simple email format check
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# keeps authentication responses in one consistent format
def create_auth_result(success, message, user=None):
    return {
        "success": success,
        "message": message,
        "user": user,
    }


# checks whether a username meets the basic account rules
def validate_username(username):
    clean_username = normalise_username(username)

    if not clean_username:
        return "A username is required."

    if len(clean_username) < MINIMUM_USERNAME_LENGTH:
        return (
            "The username must contain at least "
            f"{MINIMUM_USERNAME_LENGTH} characters."
        )

    if len(clean_username) > MAXIMUM_USERNAME_LENGTH:
        return (
            "The username cannot contain more than "
            f"{MAXIMUM_USERNAME_LENGTH} characters."
        )

    # reject spaces and special characters
    if not USERNAME_PATTERN.fullmatch(clean_username):
        return (
            "The username can contain only letters, "
            "numbers and underscores."
        )

    return ""


# removes fields that should never be returned to routes or templates
def remove_private_user_fields(user):
    if not isinstance(user, dict):
        return None

    safe_user = dict(user)

    # never expose the stored password hash
    safe_user.pop("password_hash", None)

    return safe_user


# checks the normalised email against the allowed email pattern
def is_valid_email(email):
    clean_email = normalise_email(email)

    if not clean_email:
        return False

    return EMAIL_PATTERN.fullmatch(clean_email) is not None


# checks the password length before hashing or storing it
def validate_password(password):
    clean_password = str(password or "")

    if not clean_password:
        return "A password is required."

    if len(clean_password) < MINIMUM_PASSWORD_LENGTH:
        return (
            "The password must contain at least "
            f"{MINIMUM_PASSWORD_LENGTH} characters."
        )

    if len(clean_password) > MAXIMUM_PASSWORD_LENGTH:
        return (
            "The password cannot contain more than "
            f"{MAXIMUM_PASSWORD_LENGTH} characters."
        )

    return ""


# validates registration input and creates the new account
def register_user(
    username,
    email,
    password,
    confirm_password,
    display_name="",
):
    # normalise account identifiers before validation
    clean_username = normalise_username(username)
    clean_email = normalise_email(email)

    # keep passwords as entered apart from converting empty values to strings
    clean_password = str(password or "")
    clean_confirmation = str(confirm_password or "")

    # display names may contain spaces, so only trim the ends
    clean_display_name = str(display_name or "").strip()

    # validate the username first
    username_error = validate_username(clean_username)

    if username_error:
        return create_auth_result(
            False,
            username_error,
        )

    # reject badly formatted email addresses
    if not is_valid_email(clean_email):
        return create_auth_result(
            False,
            "Please enter a valid email address.",
        )

    # stop unusually long display names before database insertion
    if len(clean_display_name) > MAXIMUM_DISPLAY_NAME_LENGTH:
        return create_auth_result(
            False,
            "The display name cannot contain more than "
            f"{MAXIMUM_DISPLAY_NAME_LENGTH} characters.",
        )

    # check password length rules before comparing confirmation
    password_error = validate_password(clean_password)

    if password_error:
        return create_auth_result(
            False,
            password_error,
        )

    # both password fields must match exactly
    if clean_password != clean_confirmation:
        return create_auth_result(
            False,
            "The passwords do not match.",
        )

    try:
        # usernames must be unique
        if username_exists(clean_username):
            return create_auth_result(
                False,
                "This username is already in use.",
            )

        # email addresses must also be unique
        if email_exists(clean_email):
            return create_auth_result(
                False,
                "An account with this email already exists.",
            )

        # store only the password hash, never the plain password
        password_hash = hash_password(clean_password)

        user = create_user(
            username=clean_username,
            email=clean_email,
            password_hash=password_hash,
            display_name=clean_display_name,
        )

        # return the user without private password data
        return create_auth_result(
            True,
            "Your account was created successfully.",
            remove_private_user_fields(user),
        )

    # validation problems raised by the database layer are shown to the user
    except ValueError as error:
        return create_auth_result(
            False,
            str(error),
        )

    # database failures use a general message rather than exposing details
    except sqlite3.Error:
        return create_auth_result(
            False,
            "The account could not be created. Please try again.",
        )


# checks login details against the stored account
def authenticate_user(username, password):
    clean_username = normalise_username(username)
    clean_password = str(password or "")

    # use the same message for missing users and wrong passwords
    # so account existence is not revealed
    generic_error = "The username or password is incorrect."

    if not clean_username or not clean_password:
        return create_auth_result(
            False,
            generic_error,
        )

    try:
        # retrieve the account using the normalised username
        user = get_user_by_username(clean_username)

    except sqlite3.Error:
        return create_auth_result(
            False,
            "Login is temporarily unavailable. Please try again.",
        )

    # unknown usernames use the same generic login error
    if user is None:
        return create_auth_result(
            False,
            generic_error,
        )

    # compare the entered password with the stored hash
    password_is_correct = verify_password(
        clean_password,
        user.get("password_hash"),
    )

    if not password_is_correct:
        return create_auth_result(
            False,
            generic_error,
        )

    # inactive accounts cannot sign in even with the correct password
    if not bool(user.get("is_active")):
        return create_auth_result(
            False,
            "This account is currently inactive.",
        )

    # successful responses also remove the password hash
    return create_auth_result(
        True,
        "Login successful.",
        remove_private_user_fields(user),
    )