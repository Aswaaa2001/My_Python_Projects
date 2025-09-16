import os
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Length

# ----------------- App Config -----------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret")
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///tasks.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ----------------- Database Model -----------------
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    done = db.Column(db.Boolean, default=False)

# ----------------- Forms -----------------
class TaskForm(FlaskForm):
    title = StringField("Title", validators=[DataRequired(), Length(max=120)])
    description = TextAreaField("Description")
    done = BooleanField("Done")
    submit = SubmitField("Save")

# ----------------- Routes -----------------
@app.route("/")
def index():
    tasks = Task.query.order_by(Task.id.desc()).all()
    return render_template("index.html", tasks=tasks)

@app.route("/add", methods=["GET", "POST"])
def add_task():
    form = TaskForm()
    if form.validate_on_submit():
        task = Task(title=form.title.data, description=form.description.data, done=form.done.data)
        db.session.add(task)
        db.session.commit()
        flash("Task added!", "success")
        return redirect(url_for("index"))
    return render_template("form.html", form=form, action="Add Task")

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_task(id):
    task = Task.query.get_or_404(id)
    form = TaskForm(obj=task)
    if form.validate_on_submit():
        task.title = form.title.data
        task.description = form.description.data
        task.done = form.done.data
        db.session.commit()
        flash("Task updated!", "info")
        return redirect(url_for("index"))
    return render_template("form.html", form=form, action="Edit Task")

@app.route("/delete/<int:id>", methods=["POST"])
def delete_task(id):
    task = Task.query.get_or_404(id)
    db.session.delete(task)
    db.session.commit()
    flash("Task deleted!", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all() 
    app.run(debug=True)
