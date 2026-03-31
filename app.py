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
# DEBUG START
# =============================
print("🚀 App starting...")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print("📁 BASE_DIR:", BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_pipeline.joblib")
print("📦 MODEL_PATH:", MODEL_PATH)
print("📦 Model exists:", os.path.exists(MODEL_PATH))


# =============================
# APP CONFIG
# =============================
app = Flask(__name__, template_folder="web/templates")

app.secret_key = os.getenv("SECRET_KEY", "fallback_secret")
print("🔑 SECRET_KEY loaded")


DATABASE_URL = os.getenv("DATABASE_URL")
print("🌐 Raw DATABASE_URL:", DATABASE_URL)

if not DATABASE_URL:
    print("⚠️ No DATABASE_URL found, using SQLite fallback")
    DATABASE_URL = "sqlite:///" + os.path.join(BASE_DIR, "app.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

print("✅ Final DATABASE_URL:", DATABASE_URL)


app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# =============================
# DB INIT
# =============================
try:
    db = SQLAlchemy(app)
    print("✅ Database initialized")
except Exception as e:
    print("❌ Database init failed:", e)


login_manager = LoginManager(app)
login_manager.login_view = "login"
print("✅ Login manager initialized")


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
# CREATE DB TABLES
# =============================
try:
    with app.app_context():
        db.create_all()
        print("✅ Tables created")

        if not User.query.filter_by(email="admin@gmail.com").first():
            admin_user = User(
                name="Admin User",
                email="admin@gmail.com",
                password_hash=generate_password_hash("admin123"),
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Admin user created")
        else:
            print("ℹ️ Admin already exists")

except Exception as e:
    print("❌ DB setup failed:", e)


# =============================
# LOAD MODEL
# =============================
try:
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print("❌ Model file NOT found")
        pipeline = None
except Exception as e:
    print("❌ Model loading failed:", e)
    pipeline = None


# =============================
# ROUTES (minimal test first)
# =============================
@app.route("/")
def index():
    print("🌐 Index page loaded")

    template_path = os.path.join(BASE_DIR, "web", "templates", "index.html")
    print("🔍 Template path:", template_path)
    print("📄 Template exists:", os.path.exists(template_path))

    return render_template("index.html")
# =============================
# AUTH ROUTES
# =============================

@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/register")
def register():
   return render_template("register.html")


# =============================
# CONTACT ROUTE
# =============================

@app.route("/contact")
def contact():
    return render_template("contact.html")
# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)