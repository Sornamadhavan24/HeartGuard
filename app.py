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
# CONFIG
# =============================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_pipeline.joblib")


app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "fallback_secret")


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Please configure it in Render.")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# =============================
# DATABASE MODELS
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
    user = db.relationship("User", backref=db.backref("login_history", lazy=True))


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
    user = db.relationship("User", backref=db.backref("feedbacks", lazy=True))


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    message = db.Column(db.Text, nullable=False)
    is_admin_reply = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship("User", backref=db.backref("chat_messages", lazy=True))


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()
    # Seed the requested default admin user
    if not User.query.filter_by(email="admin@gmail.com").first():
        admin_user = User(
            name="Admin User",
            email="admin@gmail.com",
            password_hash=generate_password_hash("admin123"),
            is_admin=True
        )
        db.session.add(admin_user)
        db.session.commit()


# =============================
# LOAD MODEL
# =============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)


# =============================
# LOGICAL RISK SCORING ENGINE
# =============================
def calculate_risk_score(form):

    score = 0

    def safe_float(value):
        try:
            return float(value)
        except:
            return 0

    age = safe_float(form.get("age"))
    bp = safe_float(form.get("trestbps"))
    chol = safe_float(form.get("chol"))
    thalach = safe_float(form.get("thalach"))
    oldpeak = safe_float(form.get("oldpeak"))
    smoker = safe_float(form.get("Smoker"))
    stress = safe_float(form.get("Stress_Level"))
    exercise = safe_float(form.get("Exercise"))
    step_count = safe_float(form.get("step_count_stress"))
    cp = safe_float(form.get("cp"))
    sudden_weight_loss = safe_float(form.get("Sudden_Weight_Loss"))

    # AGE
    if age > 60:
        score += 2
    elif age > 45:
        score += 1

    # BP
    if bp > 180:
        score += 3
    elif bp > 140:
        score += 2
    elif bp > 120:
        score += 1

    # CHOLESTEROL
    if chol > 300:
        score += 3
    elif chol > 240:
        score += 2
    elif chol > 200:
        score += 1

    # HEART RATE
    if thalach < 90:
        score += 2

    # OLDPEAK
    if oldpeak > 3:
        score += 3
    elif oldpeak > 2:
        score += 2
    elif oldpeak > 1:
        score += 1

    # SMOKER
    if smoker == 1:
        score += 2

    # STRESS
    if stress > 8:
        score += 3
    elif stress > 6:
        score += 2
    elif stress > 4:
        score += 1

    # EXERCISE (Low increases risk)
    if exercise < 1:
        score += 2
    elif exercise < 3:
        score += 1

    # STEP COUNT (Low increases risk)
    if step_count < 3000:
        score += 2
    elif step_count < 6000:
        score += 1

    # CHEST PAIN TYPE
    if cp == 3:
        score += 3
    elif cp == 2:
        score += 2
    elif cp == 1:
        score += 1

    # SUDDEN WEIGHT LOSS
    if sudden_weight_loss == 1:
        score += 2

    return score


# =============================
# ADVICE ENGINE
# =============================
def build_advice(risk, form):

    lifestyle_tips = []

    def safe_float(value):
        try:
            return float(value)
        except:
            return 0
            
    def safe_str(value):
        return str(value).lower() if value else ""

    age = safe_float(form.get("age"))
    bp = safe_float(form.get("trestbps"))
    chol = safe_float(form.get("chol"))
    smoker = safe_float(form.get("Smoker"))
    stress = safe_float(form.get("Stress_Level"))
    exercise = safe_float(form.get("Exercise"))
    step_count = safe_float(form.get("step_count_stress"))
    alcohol = safe_float(form.get("Alcohol"))
    food = safe_str(form.get("Food"))

    # Dynamic inputs evaluation
    if food == "high-fat":
        lifestyle_tips.append("Avoid high-fat foods and switch to a balanced, heart-healthy diet.")
    elif food == "low-salt":
        lifestyle_tips.append("Maintain your low-salt diet to keep blood pressure under control.")
    else:
        lifestyle_tips.append("Keep up a balanced diet to support optimal heart health.")

    if exercise < 2:
        lifestyle_tips.append("Increase your weekly exercise. Aim for at least 150 minutes of moderate activity.")
    else:
        lifestyle_tips.append("Excellent exercise habits. Continue maintaining your active lifestyle.")

    if smoker == 1:
        lifestyle_tips.append("Quit smoking immediately. This is critical for lowering cardiovascular risk.")
    else:
        lifestyle_tips.append("Continue to avoid smoking and secondhand smoke environments.")

    if alcohol == 1:
        lifestyle_tips.append("Reduce alcohol consumption to improve overall heart health.")

    if stress > 6:
        lifestyle_tips.append("Practice stress-reduction techniques daily, like meditation or yoga.")
    else:
        lifestyle_tips.append("Your stress levels are well managed today. Keep focusing on mental well-being.")

    if step_count < 6000:
        lifestyle_tips.append("Increase your daily step count to at least 8000 steps for better cardiac output.")

    if bp > 130:
        lifestyle_tips.append("Monitor your blood pressure regularly and strictly limit sodium intake.")

    if chol > 200:
        lifestyle_tips.append("Adopt a diet low in saturated fats to help lower your cholesterol levels.")

    if age > 50:
        lifestyle_tips.append("Due to age, ensure you get regular comprehensive heart health checkups.")

    # General fallback tips if we don't have enough
    general_tips = [
        "Drink at least 8 glasses of water daily for hydration.",
        "Ensure you get 7-8 hours of quality sleep every night.",
        "Add more fiber to your diet with fresh fruits and vegetables.",
        "Limit your intake of processed foods and added sugars.",
        "Consider an annual preventative health checkup."
    ]
    
    for tip in general_tips:
        if tip not in lifestyle_tips:
            lifestyle_tips.append(tip)

    # Return EXACTLY 8 instructions
    final_lifestyle = lifestyle_tips[:8]

    # Doctor suggestions remain static based on risk 
    doctor = {
        "LOW": [
            "Routine yearly checkup.",
            "Monitor cholesterol.",
            "Blood sugar yearly.",
            "Consult if symptoms appear.",
            "Family history screening."
        ],
        "MEDIUM": [
            "Consult within 1 month.",
            "Lipid profile required.",
            "ECG if needed.",
            "BP management plan.",
            "Diet consultation."
        ],
        "HIGH": [
            "Consult cardiologist within 72 hours.",
            "ECG required.",
            "Medication review.",
            "Lipid profile mandatory.",
            "Emergency if chest pain."
        ],
        "VERY HIGH": [
            "Immediate cardiologist visit.",
            "Full cardiac screening.",
            "ECG + ECHO urgently.",
            "Strict medical supervision.",
            "Emergency care immediately."
        ]
    }
    
    return final_lifestyle, doctor.get(risk, [])


# =============================
# ROUTES
# =============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/home")
@login_required
def home_page():
    return render_template("home.html")


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/contact")
@login_required
def contact():
    return render_template("contact.html")


@app.route("/feedback", methods=["GET", "POST"])
@login_required
def feedback():
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            new_feedback = Feedback(user_id=current_user.id, message=message)
            db.session.add(new_feedback)
            db.session.commit()
            flash("Thank you! Your feedback has been submitted.", "success")
            return redirect(url_for("feedback"))
        else:
            flash("Feedback message cannot be empty.", "error")
    return render_template("feedback.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            flash("Email already exists")
            return redirect(url_for("register"))

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        flash("Registration successful")
        return redirect(url_for("login"))

    return render_template("auth.html", mode="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid credentials")
            return redirect(url_for("login"))

        login_user(user)
        
        # Log login history
        history = LoginHistory(user_id=user.id)
        db.session.add(history)
        db.session.commit()

        if user.is_admin:
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("home_page"))

    return render_template("auth.html", mode="login")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    last_record = Prediction.query.filter_by(user_id=current_user.id).first()
    return render_template("dashboard.html", has_prediction=last_record is not None)


@app.route("/history")
@login_required
def history():
    records = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(Prediction.created_at.desc()).all()

    return render_template("history.html", records=records)


@app.route("/result")
@login_required
def result_page():
    # Fetch only the latest prediction for the user
    record = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(Prediction.created_at.desc()).first()

    if not record:
        flash("No prediction found. Please perform a risk assessment first.")
        return redirect(url_for("dashboard"))

    user_data = json.loads(record.inputs_json)
    tips, doctor = build_advice(record.predicted_risk, user_data)

    return render_template(
        "result.html",
        risk=record.predicted_risk,
        probability=record.probability,
        tips=tips,
        doctor=doctor,
        user_data=user_data,
        now=record.created_at
    )


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("Unauthorized access!")
        return redirect(url_for("dashboard"))

    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    
    # Get all users with their prediction counts and last login
    users_data = []
    users = User.query.all()
    for user in users:
        pred_count = Prediction.query.filter_by(user_id=user.id).count()
        last_login = LoginHistory.query.filter_by(user_id=user.id).order_by(LoginHistory.login_time.desc()).first()
        users_data.append({
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "pred_count": pred_count,
            "last_login": last_login.login_time if last_login else "Never"
        })

    # Recent login history globally
    recent_logins = LoginHistory.query.order_by(LoginHistory.login_time.desc()).limit(10).all()

    # Detailed prediction history for all users
    prediction_history = db.session.query(Prediction, User).join(User, Prediction.user_id == User.id).order_by(Prediction.created_at.desc()).all()

    # Get all feedbacks
    feedbacks = db.session.query(Feedback, User).join(User, Feedback.user_id == User.id).order_by(Feedback.created_at.desc()).all()

    # Chat data for admin
    # Get all distinct users who have messaged
    messaged_users = db.session.query(User).join(ChatMessage, User.id == ChatMessage.user_id).distinct().all()
    
    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        users_data=users_data,
        recent_logins=recent_logins,
        prediction_history=prediction_history,
        feedbacks=feedbacks,
        messaged_users=messaged_users
    )


# =============================
    # CHAT ROUTES
# =============================
@app.route("/send_message", methods=["POST"])
@login_required
def send_message():
    msg_text = request.json.get("message")
    if not msg_text:
        return {"status": "error", "message": "Empty message"}, 400
    
    new_msg = ChatMessage(user_id=current_user.id, message=msg_text, is_admin_reply=False)
    db.session.add(new_msg)
    db.session.commit()
    return {"status": "success"}


@app.route("/get_messages")
@login_required
def get_messages():
    target_user_id = request.args.get("user_id") if current_user.is_admin else current_user.id
    if not target_user_id:
        target_user_id = current_user.id
        
    messages = ChatMessage.query.filter_by(user_id=target_user_id).order_by(ChatMessage.created_at.asc()).all()
    return {
        "messages": [
            {
                "message": m.message,
                "is_admin_reply": m.is_admin_reply,
                "created_at": m.created_at.strftime("%H:%M")
            } for m in messages
        ]
    }


@app.route("/admin/reply_message", methods=["POST"])
@login_required
def admin_reply():
    if not current_user.is_admin:
        return {"status": "error", "message": "Unauthorized"}, 403
    
    user_id = request.json.get("user_id")
    msg_text = request.json.get("message")
    
    if not user_id or not msg_text:
        return {"status": "error", "message": "Invalid data"}, 400
        
    new_msg = ChatMessage(user_id=user_id, message=msg_text, is_admin_reply=True)
    db.session.add(new_msg)
    db.session.commit()
    return {"status": "success"}


@app.route("/predict", methods=["POST"])
@login_required
def predict():

    # Validation: Ensure all content is filled
    required_fields = [
        'name', 'age', 'sex', 'Smoker', 'Exercise', 'Stress_Level', 
        'step_count_stress', 'Alcohol', 'Food', 'Sudden_Weight_Loss', 
        'cp', 'trestbps', 'chol', 'thalach', 'oldpeak'
    ]
    
    for field in required_fields:
        if not request.form.get(field):
            flash("All fields are mandatory. Please fill in the complete health profile.", "error")
            return redirect(url_for('dashboard'))

    expected_cols = list(pipeline.named_steps["preprocess"].feature_names_in_)
    data_dict = {col: np.nan for col in expected_cols}

    for col in expected_cols:
        val = request.form.get(col)
        if val:
            try:
                data_dict[col] = float(val)
            except:
                data_dict[col] = val

    X_new = pd.DataFrame([data_dict])

    # ML Prediction (optional)
    ml_pred = pipeline.predict(X_new)[0]

    # Logical Severity Prediction and Percentage Mapping
    risk_score = calculate_risk_score(request.form)

    # Convert risk score to a percentage based on ranges:
    # low risk : 0 to 45
    # medium risk: 46 to 60
    # high risk: 61 to 85
    # very high risk: 86 above
    if risk_score <= 3:
        risk = "LOW"
        probability = round((risk_score / 3.0) * 45, 2)
    elif risk_score <= 6:
        risk = "MEDIUM"
        probability = round(46 + ((risk_score - 3) / 3.0) * 14, 2)
    elif risk_score <= 10:
        risk = "HIGH"
        probability = round(61 + ((risk_score - 6) / 4.0) * 24, 2)
    else:
        risk = "VERY HIGH"
        probability = round(min(100.0, 86 + (risk_score - 10) * 2), 2)

    tips, doctor = build_advice(risk, request.form)

    record = Prediction(
        user_id=current_user.id,
        inputs_json=json.dumps(request.form.to_dict()),
        predicted_risk=risk,
        probability=probability
    )
    db.session.add(record)
    db.session.commit()

    return redirect(url_for("result_page"))


if __name__ == "__main__":
    # Use host="0.0.0.0" to allow other computers on your network to connect
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)