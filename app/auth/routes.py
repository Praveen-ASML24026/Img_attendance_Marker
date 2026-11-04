from flask import render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user

from app.auth import auth
from app.models import User
from app.extensions import db


@auth.route("/", methods=["GET", "POST"])
@auth.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("students.list_students"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("students.list_students"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@auth.route("/register", methods=["GET", "POST"])
def register():
    """Teacher Account Registration Flow."""
    if current_user.is_authenticated:
        return redirect(url_for("students.list_students"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password or not confirm_password:
            flash("All fields are required.", "error")
            return redirect(url_for("auth.register"))

        if len(username) < 3:
            flash("Username must contain at least 3 characters.", "error")
            return redirect(url_for("auth.register"))

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
            return redirect(url_for("auth.register"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("auth.register"))

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("Username already exists. Please choose a different username or sign in.", "error")
            return redirect(url_for("auth.register"))

        new_teacher = User(username=username)
        new_teacher.set_password(password)

        try:
            db.session.add(new_teacher)
            db.session.commit()
            flash("Account created successfully! You can now sign in.", "success")
            return redirect(url_for("auth.login"))
        except Exception as e:
            db.session.rollback()
            flash(f"Error creating account: {str(e)}", "error")
            return redirect(url_for("auth.register"))

    return render_template("register.html")


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
