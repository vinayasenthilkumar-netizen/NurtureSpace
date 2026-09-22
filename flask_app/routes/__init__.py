# used to preserve the original route name when applying custom decorators
from functools import wraps
# flask helpers used by the authentication decorator
from flask import flash, redirect, session, url_for
# privacy helpers are used to check whether bookmark saving is currently allowed
from database.privacy import get_privacy_preferences, privacy_is_current
# small helper used by routes that only need to know whether a user is signed in
def has_account_session():
    return bool(session.get("user_id"))


# creates a reusable login requirement for Flask routes
def account_required(message, category="warning"):
    # this outer function lets each route choose its own flash message
    def decorator(view_function):

        @wraps(view_function)
        def wrapped_view(*args, **kwargs):

            # stop the route early if there is no signed-in account
            if not has_account_session():
                flash(message, category)
                return redirect(url_for("auth.login"))

            # otherwise continue to the original route
            return view_function(*args, **kwargs)

        return wrapped_view

    return decorator


# checks the user's current privacy choice before allowing a new bookmark
def can_save_bookmarks(user_id):
    # bookmark saving always requires an authenticated account
    if not user_id:
        return False

    try:
        # load the preferences saved for this account
        preferences = get_privacy_preferences(user_id)

        # saving is allowed only when the privacy notice is current
        # and the resource-bookmark option has been enabled
        return bool(
            preferences
            and privacy_is_current(user_id)
            and preferences.get("save_bookmarked_resources", False)
        )

    except Exception as error:
        # fail safely by disabling saving if privacy settings cannot be checked
        print("Bookmark privacy check error:", error)
        return False