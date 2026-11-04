import click

from app import create_app
from app.extensions import db
from app.models import User

app = create_app()


@app.cli.command("create-teacher")
def create_teacher():
    username = click.prompt("Enter teacher username")

    if User.query.filter_by(username=username).first():
        click.echo("Username already exists.")
        return

    password = click.prompt("Enter password", hide_input=True)
    confirmation = click.prompt("Confirm password", hide_input=True)

    if password != confirmation:
        click.echo("Passwords do not match.")
        return

    if len(password) < 12:
        click.echo("Password must contain at least 12 characters.")
        return

    teacher = User(username=username)
    teacher.set_password(password)

    db.session.add(teacher)
    db.session.commit()

    click.echo("Teacher account created successfully.")


if __name__ == "__main__":
    app.run(debug=True)