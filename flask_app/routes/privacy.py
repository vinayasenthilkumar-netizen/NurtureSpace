
# flask helpers for routing, templates, forms, sessions and redirects
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# current version of the privacy notice accepted by the user
from config.settings import CURRENT_PRIVACY_VERSION

# database helpers for loading and saving account privacy choices
from database.privacy import (
    get_privacy_preferences,
    save_privacy_preferences,
)

# shared decorator used to protect account-only routes
from flask_app.routes import account_required


# privacy routes do not need an additional URL prefix
privacy_blueprint = Blueprint("privacy", __name__)


def render_privacy(preferences, first_setup=None):
    # when not supplied directly, having no saved preferences means
    # the user is completing privacy setup for the first time
    if first_setup is None:
        first_setup = preferences is None

    # use an empty dictionary when no preferences have been stored yet
    return render_template(
        "privacy.html",
        first_setup=first_setup,
        preferences=preferences or {},
    )


@privacy_blueprint.route("/privacy", methods=["GET", "POST"])
@account_required(
    "Please log in before viewing privacy information.",
    "error",
)
def privacy_settings():
    # account_required ensures this id is available for the current user
    user_id = session["user_id"]

    try:
        # load the user's existing choices before displaying or updating them
        existing_preferences = get_privacy_preferences(user_id)

    except Exception:
        # keep the privacy page usable even if the existing record cannot be loaded
        existing_preferences = None

        flash(
            "Your privacy information could not be loaded.",
            "error",
        )

    # no existing record means the user has not completed privacy setup yet
    first_setup = existing_preferences is None

    # GET only displays the currently stored privacy choices
    if request.method == "GET":
        return render_privacy(
            existing_preferences,
            first_setup=first_setup,
        )


    # read each privacy option independently so no choice is enabled automatically
    submitted_preferences = {
        "save_approved_checkins":
            request.form.get("save_approved_checkins") == "on",

        "use_saved_history_for_personalisation":
            request.form.get(
                "use_saved_history_for_personalisation"
            ) == "on",

        "save_bookmarked_resources":
            request.form.get("save_bookmarked_resources") == "on",
    }


    # the privacy notice must be acknowledged before any choices are stored
    if request.form.get("understands_privacy") != "on":
        flash(
            "Please confirm that you understand how your "
            "information will be handled.",
            "error",
        )

        # preserve the options the user just selected when showing the page again
        return render_privacy(
            submitted_preferences,
            first_setup=first_setup,
        )


    try:
        # save all privacy choices together with the current privacy-policy version
        preferences = save_privacy_preferences(
            user_id=user_id,

            # controls whether approved check-ins may be persisted
            save_approved_checkins=(
                submitted_preferences[
                    "save_approved_checkins"
                ]
            ),

            # controls whether saved history may be used as Assistant context
            use_saved_history_for_personalisation=(
                submitted_preferences[
                    "use_saved_history_for_personalisation"
                ]
            ),

            # controls whether curated resource bookmarks may be persisted
            save_bookmarked_resources=(
                submitted_preferences[
                    "save_bookmarked_resources"
                ]
            ),

            # records which privacy notice version these choices belong to
            privacy_version=CURRENT_PRIVACY_VERSION,
        )

    except ValueError as error:
        # show validation messages returned intentionally by the privacy service
        flash(str(error), "error")

        return render_privacy(
            submitted_preferences,
            first_setup=first_setup,
        )

    except Exception:
        # unexpected persistence errors should not discard the submitted selections
        flash(
            "Your privacy choices could not be saved. "
            "Please try again.",
            "error",
        )

        return render_privacy(
            submitted_preferences,
            first_setup=first_setup,
        )


    # keep the active session aligned with the preferences now stored in the database
    session.update({
        # retain the complete preference record for routes that need it
        "privacy_preferences": preferences,

        # successful privacy acknowledgement completes consent setup
        "consent_accepted": True,
        "privacy_reminder_seen": True,
        "privacy_setup_required": False,

        # individual values are also stored for simple permission checks elsewhere
        "save_approved_checkins": (
            preferences["save_approved_checkins"]
        ),

        "use_saved_history_for_personalisation": (
            preferences[
                "use_saved_history_for_personalisation"
            ]
        ),

        "save_bookmarked_resources": (
            preferences["save_bookmarked_resources"]
        ),
    })

    # confirm that the choices were successfully persisted
    flash(
        "Your privacy choices have been saved.",
        "success",
    )

    # after setup or an update, return the user to the main Today page
    return redirect(url_for("home.home"))