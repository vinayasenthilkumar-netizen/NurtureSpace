# operating system helpers are used to build the session storage path
import os

# session lifetime is stored as a timedelta value
from datetime import timedelta

# filesystem cache is used to keep Flask session data on the server
from cachelib.file import FileSystemCache

# main Flask application and server-side session extension
from flask import Flask
from flask_session import Session

# create the required database tables when the application starts
from database.schema import initialise_database


# create and configure the Flask application
def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # main application settings
    app.config.update(
        SECRET_KEY = "my-dev-key",
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,
    )

    # store Flask session data on the server instead of inside the browser cookie
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    session_folder = os.path.join(project_root, "flask_session")

    # keep session files for up to 12 hours
    session_cache = FileSystemCache(
        cache_dir=session_folder,
        threshold=500,
        default_timeout=12 * 60 * 60,
    )

    # configure server-side sessions and safer cookie settings
    app.config.update(
        SESSION_TYPE="cachelib",
        SESSION_CACHELIB=session_cache,
        SESSION_PERMANENT=False,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    # attach Flask-Session to the application
    Session(app)

    # make sure the database schema is ready before routes are used
    initialise_database()

    # import blueprints here after the app has been configured
    from flask_app.routes.home import home_blueprint
    from flask_app.routes.auth import auth_blueprint
    from flask_app.routes.privacy import privacy_blueprint
    from flask_app.routes.questions import questions_blueprint
    from flask_app.routes.reflection import reflection_blueprint
    from flask_app.routes.review import review_blueprint
    from flask_app.routes.dashboard import dashboard_blueprint
    from flask_app.routes.assistant import assistant_blueprint
    from flask_app.routes.resources import resources_blueprint
    from flask_app.routes.history import history_blueprint
    from flask_app.routes.settings import settings_blueprint

    # keep all application blueprints together for registration
    blueprints = (
        home_blueprint,
        auth_blueprint,
        privacy_blueprint,
        questions_blueprint,
        reflection_blueprint,
        review_blueprint,
        dashboard_blueprint,
        assistant_blueprint,
        resources_blueprint,
        history_blueprint,
        settings_blueprint,
    )

    # register each route group with the Flask app
    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    return app