
# flask helpers for routing, templates, form data, sessions and redirects
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# database functions for reading curated resources and managing user bookmarks
from database.resources import (
    bookmark_resource,
    get_bookmarked_resources,
    get_resource_by_id,
    remove_resource_bookmark,
)

# shared authentication guard and privacy check for bookmark saving
from flask_app.routes import account_required, can_save_bookmarks


# all resource routes are grouped under /resources
resources_blueprint = Blueprint(
    "resources",
    __name__,
    url_prefix="/resources",
)


# saved-resource pages require a signed-in account
login_required = account_required(
    "Please log in to view saved resources.",
    "warning",
)


def resource_return_url():
    # forms can request a return to Today or the Saved Resources page
    return_to = str(
        request.form.get("return_to", "saved") or "saved"
    ).strip().lower()

    # only the expected internal value is allowed to route back to Today
    if return_to == "today":
        return url_for("home.home")

    # every other value safely falls back to Saved Resources
    return url_for("resources.saved_resources")


def _resource_id_from_form():
    # normalise the submitted value before using it in a database lookup
    return str(
        request.form.get("resource_id", "") or ""
    ).strip()


@resources_blueprint.route("/saved")
@login_required
def saved_resources():
    # bookmarks always belong to the currently authenticated account
    user_id = session.get("user_id")

    try:
        # load only resources this user has previously bookmarked
        resources = get_bookmarked_resources(user_id)

    except Exception as error:
        # keep the page available even if the bookmark query fails
        print("Saved Resources load error:", error)
        resources = []

        flash(
            "Your saved resources could not be loaded.",
            "error",
        )

    # use display name first, then username for the sidebar greeting
    welcome_name = (
        session.get("display_name")
        or session.get("username")
        or "there"
    )

    # render both the bookmark list and the current saving permission
    return render_template(
        "saved_resources.html",
        saved_resources=resources,
        saved_resource_count=len(resources),
        welcome_name=welcome_name,
        bookmark_saving_allowed=can_save_bookmarks(user_id),
    )


@resources_blueprint.route("/save", methods=["POST"])
@login_required
def save_resource():
    user_id = session.get("user_id")

    # resource id identifies which curated item should be bookmarked
    resource_id = _resource_id_from_form()

    # return destination depends on whether saving started from Today or Saved Resources
    return_url = resource_return_url()


    # stop early if the form did not contain a valid resource id
    if not resource_id:
        flash(
            "The resource could not be identified.",
            "warning",
        )
        return redirect(return_url)


    # privacy preference must explicitly allow bookmark persistence
    if not can_save_bookmarks(user_id):
        flash(
            "Saving resources is turned off in your privacy settings.",
            "warning",
        )

        # direct the user to privacy controls rather than silently enabling saving
        return redirect(url_for("privacy.privacy_settings"))


    try:
        # confirm that the curated resource still exists before bookmarking it
        resource = get_resource_by_id(resource_id)

    except Exception as error:
        # database lookup failure is treated the same as an unavailable resource
        print("Resource validation error:", error)
        resource = None


    # do not create bookmarks for resources that no longer exist
    if not resource:
        flash(
            "That resource is no longer available.",
            "warning",
        )
        return redirect(return_url)


    try:
        # create the user-to-resource bookmark relationship
        bookmark_resource(
            user_id=user_id,
            resource_id=resource_id,
        )

    except Exception as error:
        # bookmarking failure should not interrupt the rest of the application
        print("Resource bookmark error:", error)

        flash(
            "The resource could not be saved.",
            "error",
        )
        return redirect(return_url)


    # return to the page the user was originally browsing
    flash("Resource saved.", "success")
    return redirect(return_url)


@resources_blueprint.route("/remove", methods=["POST"])
@login_required
def remove_resource():

    user_id = session.get("user_id")
    resource_id = _resource_id_from_form()
    return_url = resource_return_url()


    # a bookmark cannot be removed without identifying the resource
    if not resource_id:
        flash(
            "The resource could not be identified.",
            "warning",
        )
        return redirect(return_url)


    # removal does not check can_save_bookmarks()
    # this lets users clean up existing bookmarks even after disabling saving
    try:
        remove_resource_bookmark(
            user_id=user_id,
            resource_id=resource_id,
        )

    except Exception as error:
        # removing a bookmark must not remove or modify the shared curated resource
        print("Remove saved resource error:", error)

        flash(
            "The saved resource could not be removed.",
            "error",
        )
        return redirect(return_url)


    # only the user's bookmark relationship has been removed
    flash(
        "Resource removed from Saved Resources.",
        "success",
    )

    return redirect(return_url)