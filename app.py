# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash


# =============================
# APP CONFIG
# =============================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)

app.secret_key = os.getenv("SECRET_KEY", "fallback_secret")

# =============================
# DATABASE CONFIG
# =============================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///" + os.path.join(BASE_DIR, "app.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# =============================
# LOGIN MANAGER
# =============================
login_manager = LoginManager(app)
login_manager.login_view = "login"


# =============================
# MODELS
# =============================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(180), unique=True)
    password_hash = db.Column(db.String(255))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class LoginHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    login_time = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    inputs_json = db.Column(db.Text)
    predicted_risk = db.Column(db.String(20))
    probability = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    message = db.Column(db.Text, nullable=False)
    is_admin_reply = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# =============================
# CREATE TABLES
# =============================
with app.app_context():
    db.create_all()

    # Create default admin
    if not User.query.filter_by(email="admin@gmail.com").first():
        admin = User(
            name="Admin",
            email="admin@gmail.com",
            password_hash=generate_password_hash("admin123"),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()


# =============================
# LOAD MODEL (Optional)
# =============================
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_pipeline.joblib")

if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
else:
    pipeline = None


# =============================
# ROUTES
# =============================

# Home
@app.route("/")
def index():
    return render_template("index.html")


# =============================
# AUTH ROUTES
# =============================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            email = request.form.get("email")
            password = request.form.get("password")

            user = User.query.filter_by(email=email).first()

            if user and check_password_hash(user.password_hash, password):
                login_user(user)

                # Save login history
                history = LoginHistory(user_id=user.id)
                db.session.add(history)
                db.session.commit()

                flash("Login successful!", "success")
                return redirect(url_for("index"))
            else:
                flash("Invalid email or password", "danger")

        except Exception as e:
            print("Login error:", e)
            flash("Something went wrong", "danger")

    return render_template("auth.html", mode="login")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            name = request.form.get("name")
            email = request.form.get("email")
            password = request.form.get("password")

            if User.query.filter_by(email=email).first():
                flash("Email already exists", "warning")
                return redirect(url_for("register"))

            new_user = User(
                name=name,
                email=email,
                password_hash=generate_password_hash(password)
            )

            db.session.add(new_user)
            db.session.commit()

            flash("Registration successful!", "success")
            return redirect(url_for("login"))

        except Exception as e:
            print("Register error:", e)
            flash("Something went wrong", "danger")

    return render_template("auth.html", mode="register")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully", "info")
    return redirect(url_for("index"))


# =============================
# CONTACT
# =============================
@app.route("/contact", methods=["GET", "POST"])
def contact():
    try:
        if request.method == "POST":
            message = request.form.get("message")

            if current_user.is_authenticated:
                feedback = Feedback(
                    user_id=current_user.id,
                    message=message
                )
                db.session.add(feedback)
                db.session.commit()

                flash("Message sent!", "success")
            else:
                flash("Please login to send message", "warning")

        return render_template("contact.html")

    except Exception as e:
        print("Contact error:", e)
        return "Error loading contact page"


# =============================
# PROTECTED DASHBOARD (optional)
# =============================
@app.route("/dashboard")
@login_required
def dashboard():
    return f"Welcome {current_user.name}!"


# =============================
# RUN
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)