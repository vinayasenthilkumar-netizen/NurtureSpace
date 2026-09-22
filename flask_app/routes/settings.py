
# flask imports used for routing, templates, sessions and form handling
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# database helper used to load the currently signed-in user
from database.users import get_user_by_id

# shared route guard used to require an authenticated account
from flask_app.routes import account_required

# account-service functions keep password and deletion logic outside the route file
from services.account_service import (
    change_password_with_current_password,
    delete_account_with_password,
)


# all settings routes are grouped under the /settings prefix
settings_blueprint = Blueprint(
    "settings",
    __name__,
    url_prefix="/settings",
)


# reuse the shared account guard with a settings-specific message
login_required = account_required(
    "Please log in to open Settings.",
    "info",
)


@settings_blueprint.route("/")
@login_required
def settings_page():
    # load the account using the id stored in the authenticated session
    user = get_user_by_id(session.get("user_id"))

    # if the account no longer exists, remove the stale session
    if user is None:
        session.clear()
        flash("Your account could not be found.", "error")
        return redirect(url_for("auth.login"))

    # display name is preferred, with username and a simple fallback after that
    welcome_name = (
        user.get("display_name")
        or user.get("username")
        or "there"
    )

    # pass the current account details to the settings template
    return render_template(
        "settings.html",
        user=user,
        welcome_name=welcome_name,
    )


@settings_blueprint.route("/change-password", methods=["POST"])
@login_required
def change_password():
    # the service handles validation and verifies the existing password first
    result = change_password_with_current_password(
        user_id=session.get("user_id"),
        username=session.get("username"),
        current_password=str(
            request.form.get("current_password", "")
        ),
        new_password=str(
            request.form.get("new_password", "")
        ),
        confirm_password=str(
            request.form.get("confirm_password", "")
        ),
    )

    # show either the success message or the validation/error message returned by the service
    flash(
        result["message"],
        "success" if result["success"] else "error",
    )

    # return to Settings after the password-change attempt
    return redirect(url_for("settings.settings_page"))


@settings_blueprint.route("/delete-account", methods=["POST"])
@login_required
def delete_account():
    # require the explicit confirmation checkbox before attempting deletion
    if request.form.get("confirm_delete") != "yes":
        flash(
            "Please confirm that you want to permanently delete your account.",
            "error",
        )
        return redirect(url_for("settings.settings_page"))

    # password verification and the actual database deletion are handled by the service
    result = delete_account_with_password(
        user_id=session.get("user_id"),
        username=session.get("username"),
        password=str(
            request.form.get("delete_password", "")
        ),
    )

    # keep the user signed in when deletion fails so they can correct the problem
    if not result["success"]:
        flash(result["message"], "error")
        return redirect(url_for("settings.settings_page"))

    # remove all authentication state after the account has been deleted
    session.clear()

    # confirm deletion and return the user to the login page
    flash(result["message"], "success")
    return redirect(url_for("auth.login"))