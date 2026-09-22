
# flask helpers for routes, forms, templates, sessions and redirects
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# privacy helpers are used after login to decide whether setup is still required
from database.privacy import get_privacy_preferences, privacy_is_current
# authentication and registration logic is kept in the service layer
from services.auth_service import authenticate_user, register_user
# all authentication routes are grouped under /auth
auth_blueprint = Blueprint("auth", __name__, url_prefix="/auth")


def store_account_session(user):
    # only the basic account information needed across authenticated pages
    # is copied into the current browser session
    session.update({
        "user_id": user["id"],
        "username": user["username"],
        "user_email": user["email"],
        "display_name": user.get("display_name"),
        "is_logged_in": True,
    })


def set_privacy_session(*, accepted, setup_required, preferences=None):
    # these fields let the rest of the application know whether the user
    # has acknowledged the current privacy notice
    session.update({
        "privacy_preferences": preferences,
        "consent_accepted": accepted,
        "privacy_reminder_seen": accepted,
        "privacy_setup_required": setup_required,
    })


@auth_blueprint.route("/login", methods=["GET", "POST"])
def login():
    # already authenticated users do not need to see the login form again
    if session.get("user_id"):
        return redirect(url_for("home.home"))

    # GET simply displays an empty login form
    if request.method == "GET":
        return render_template("auth/login.html", username="")

    # normalise username but keep the password exactly as entered
    username = str(request.form.get("username", "")).strip()
    password = str(request.form.get("password", ""))

    # credential checking is handled by the authentication service
    result = authenticate_user(
        username=username,
        password=password,
    )

    # failed authentication returns the user to the same form
    if not result["success"]:
        flash(result["message"], "error")

        # keep the username so the user does not need to type it again
        return render_template(
            "auth/login.html",
            username=username,
        )

    # successful authentication creates the account session
    user = result["user"]
    store_account_session(user)

    # load the saved privacy choices associated with this account
    preferences = get_privacy_preferences(user["id"])

    # users who already accepted the current privacy version
    # can continue directly into the application
    if privacy_is_current(user["id"]):
        set_privacy_session(
            accepted=True,
            setup_required=False,
            preferences=preferences,
        )

        return redirect(url_for("home.home"))

    # missing or outdated privacy acknowledgement requires review first
    set_privacy_session(
        accepted=False,
        setup_required=True,
        preferences=preferences,
    )

    flash(
        "Please review your privacy choices before continuing.",
        "info",
    )

    return redirect(url_for("privacy.privacy_settings"))


@auth_blueprint.route("/register", methods=["GET", "POST"])
def register():
    # signed-in users should not create another account in the same session
    if session.get("user_id"):
        return redirect(url_for("home.home"))

    # GET displays a clean registration form
    if request.method == "GET":
        return render_template(
            "auth/register.html",
            username="",
            display_name="",
            email="",
        )

    # collect the registration fields submitted by the user
    username = str(request.form.get("username", "")).strip()
    display_name = str(request.form.get("display_name", "")).strip()
    email = str(request.form.get("email", "")).strip()

    # passwords are passed through without stripping so the service
    # receives exactly what the user entered
    password = str(request.form.get("password", ""))
    confirm_password = str(request.form.get("confirm_password", ""))

    # validation, password handling and account creation stay in the service layer
    result = register_user(
        username=username,
        email=email,
        password=password,
        confirm_password=confirm_password,
        display_name=display_name,
    )

    # if registration fails, preserve the non-password form fields
    if not result["success"]:
        flash(result["message"], "error")

        return render_template(
            "auth/register.html",
            username=username,
            display_name=display_name,
            email=email,
        )

    # successful registration signs the new account into the current session
    store_account_session(result["user"])

    # new users must complete privacy setup before entering the application
    set_privacy_session(
        accepted=False,
        setup_required=True,
        preferences=None,
    )

    flash(
        "Your account was created successfully. "
        "Please choose your privacy preferences.",
        "success",
    )

    return redirect(url_for("privacy.privacy_settings"))


@auth_blueprint.route("/logout", methods=["GET"])
def logout():
    # remove account, privacy and temporary workflow state from this session
    session.clear()
    flash("You have been logged out.", "success")
    # return to Login after the authenticated session has been removed
    return redirect(url_for("auth.login"))