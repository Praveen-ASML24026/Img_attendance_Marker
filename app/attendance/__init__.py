from flask import Blueprint

attendance = Blueprint("attendance", __name__, url_prefix="/attendance")

from app.attendance import routes

