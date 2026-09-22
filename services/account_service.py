
# sqlite is used to catch database errors during account changes
import sqlite3
# imports user database functions for password updates and account deletion
from database.users import delete_user, update_user_password
# imports login checking and password validation helpers
from services.auth_service import authenticate_user, validate_password
# imports the password hashing helper used before storing a new password
from services.password_service import hash_password


# returns a consistent result for account-management actions
def create_account_result(success, message):

    return {
        "success": success,
        "message": message,
    }


# checks that the supplied password belongs to the logged-in account
def verify_current_account(user_id, username, password):
    #Verify the current password belongs to the logged-in account
    if not user_id or not username:
        return None, "The account could not be identified."

    if not password:
        return None, "Please enter your current password."

    # reuse the normal login check to verify the password
    authentication = authenticate_user(
        username=username,
        password=password,
    )

    if not authentication["success"]:
        return None, "The current password is incorrect."

    user = authentication.get("user")

    if not user:
        return None, "The account could not be verified."

    # make sure the authenticated account matches the logged-in user
    try:
        if int(user["id"]) != int(user_id):
            return None, "The account could not be verified."
    except (TypeError, ValueError, KeyError):
        return None, "The account could not be verified."

    return user, ""


# changes the password after checking the current account and new password
def change_password_with_current_password(
    user_id,
    username,
    current_password,
    new_password,
    confirm_password,
):

    _, error = verify_current_account(
        user_id=user_id,
        username=username,
        password=current_password,
    )

    if error:
        return create_account_result(False, error)

    # check the new password against the normal password rules
    password_error = validate_password(new_password)

    if password_error:
        return create_account_result(
            False,
            password_error,
        )
    if new_password != confirm_password:
        return create_account_result(
            False,
            "The new passwords do not match.",
        )
    if current_password == new_password:
        return create_account_result(
            False,
            "Please choose a different password.",
        )
    try:
        # hash the new password before saving it
        password_hash = hash_password(new_password)

        updated = update_user_password(
            user_id=user_id,
            password_hash=password_hash,
        )
    except (sqlite3.Error, ValueError):
        return create_account_result(
            False,
            "The password could not be changed. Please try again.",
        )
    if not updated:
        return create_account_result(
            False,
            "The password could not be changed.",
        )
    return create_account_result(
        True,
        "Your password was changed successfully.",
    )


# deletes an account only after the current password has been verified
def delete_account_with_password(user_id, username, password):
    _, error = verify_current_account(
        user_id=user_id,
        username=username,
        password=password,
    )

    if error:
        return create_account_result(False, error)

    # remove the user account after successful verification
    try:
        deleted = delete_user(user_id)
    except sqlite3.Error:
        return create_account_result(
            False,
            "The account could not be deleted. Please try again.",
        )
    if not deleted:
        return create_account_result(
            False,
            "The account could not be deleted.",
        )
    return create_account_result(
        True,
        "Your account was permanently deleted.",
    )