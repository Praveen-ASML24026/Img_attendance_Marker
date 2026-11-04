import os
import logging
from flask import Flask, render_template
from sqlalchemy import event
from sqlalchemy.engine import Engine

from config import Config
from app.extensions import db, login_manager, csrf
from app.models import User, Student, FaceEmbedding, AttendanceSession, AttendanceRecord


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Enables SQLite Write-Ahead Logging (WAL) and sets busy timeout to 30s.
    Prevents concurrency locks when inspecting with SQLite GUI tools.
    """
    if hasattr(dbapi_connection, "cursor"):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=30000;")
        except Exception:
            pass
        finally:
            cursor.close()


def create_app(test_config=None):
    flask_app = Flask(__name__, instance_relative_config=True)

    flask_app.config.from_object(Config)

    if test_config:
        flask_app.config.update(test_config)

    os.makedirs(flask_app.instance_path, exist_ok=True)

    db.init_app(flask_app)
    login_manager.init_app(flask_app)
    csrf.init_app(flask_app)

    # Blueprints
    from app.auth.routes import auth
    from app.students.routes import students
    from app.attendance import attendance

    flask_app.register_blueprint(auth)
    flask_app.register_blueprint(students)
    flask_app.register_blueprint(attendance)

    # Global Error Handlers
    @flask_app.errorhandler(404)
    def handle_not_found(error):
        return render_template("errors/404.html"), 404

    @flask_app.errorhandler(413)
    def handle_payload_too_large(error):
        return render_template("errors/413.html"), 413

    @flask_app.errorhandler(500)
    def handle_server_error(error):
        db.session.rollback()
        return render_template("errors/500.html"), 500

    # Resilient Database Table Initialization
    with flask_app.app_context():
        try:
            db.create_all()
        except Exception as e:
            flask_app.logger.warning(
                f"[DB Warning] Could not execute db.create_all() on startup: {e}. "
                "Ensure DB Browser or concurrent database locks are closed."
            )

    return flask_app
