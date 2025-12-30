from __future__ import division, print_function
# coding=utf-8

import os
import random
import sqlite3
import time

import numpy as np
import pandas as pd
import requests
import bcrypt

# Keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

# Flask utils
from flask import Flask, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)

# IMPORTANT: needed for session-based OTP storage
# In production set FLASK_SECRET_KEY in environment.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# OTP expiry (seconds)
OTP_TTL_SECONDS = 5 * 60  # 5 minutes

# Resend config (free tier is fine)
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
OTP_FROM_EMAIL = os.environ.get("OTP_FROM_EMAIL", "onboarding@resend.dev")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_db_connection():
    con = sqlite3.connect('signup.db')
    con.row_factory = sqlite3.Row
    return con


def ensure_users_table():
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            mobile TEXT,
            name TEXT
        )
    """)
    con.commit()
    con.close()


# --- Model loading / metrics ---
model_path2 = 'model.h5'

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

CTS = load_model(
    model_path2,
    custom_objects={'f1_score': f1_m, 'precision_score': precision_m, 'recall_score': recall_m},
    compile=False
)

def model_predict2(image_path, model):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    result = int(np.argmax(model.predict(image)))

    if result == 0:
        return "Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma (Bowen's disease) (AKIEC)", "result.html"
    elif result == 1:
        return "Basal cell carcinoma (BCC) is a type of skin cancer that forms in the basal cells of your skin.", "result.html"
    elif result == 2:
        return "Benign Keratosis-like Lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)", "result.html"
    elif result == 3:
        return "A dermatofibroma is a common overgrowth of the fibrous tissue situated in the dermis.", "result.html"
    elif result == 4:
        return "Melanoma is a kind of skin cancer that starts in the melanocytes.", "result.html"
    elif result == 5:
        return "Melanocytic nevus is a non-cancerous disorder of pigment-producing skin cells (moles/birthmarks).", "result.html"
    elif result == 6:
        return "Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).", "result.html"
    else:
        return "Unable to classify. Please try another image.", "result.html"


# --- Resend email sender ---
def send_otp_email_resend(to_email: str, otp: int):
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY is not set in environment variables.")

    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "from": OTP_FROM_EMAIL,
        "to": [to_email],
        "subject": "Your OTP Code",
        "html": f"<p>Your OTP is: <b>{otp}</b></p><p>This code expires in 5 minutes.</p>"
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    if resp.status_code >= 400:
        raise RuntimeError(f"Resend error {resp.status_code}: {resp.text}")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)

    if filename == "" or not allowed_file(filename):
        return "Invalid file. Please upload png/jpg/jpeg.", 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(file_path)

    pred, output_page = model_predict2(file_path, CTS)
    return render_template(output_page, pred_output=pred, img_src=UPLOAD_FOLDER + filename)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/logon')
def logon():
    return render_template('signup.html')


@app.route('/login')
def login():
    return render_template('signin.html')


@app.route('/index')
def index():
    return render_template('index.html')


# ✅ UPDATED: SIGNUP now POST (no credentials in URL)
@app.route("/signup", methods=["POST"])
def signup():
    ensure_users_table()

    username = request.form.get('user', '').strip()
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    number = request.form.get('mobile', '').strip()
    password = request.form.get('password', '')

    if not username or not email or not password:
        return "Missing required fields.", 400

    # Check duplicates (optional but useful)
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT 1 FROM info WHERE user = ? OR email = ?", (username, email))
    exists = cur.fetchone()
    con.close()
    if exists:
        return "User/email already exists. Please login or use a different email.", 400

    # ✅ Hash password with bcrypt BEFORE storing anywhere
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    otp = random.randint(1000, 9999)

    # Store in session (not globals)
    session["pending_signup"] = {
        "otp": otp,
        "otp_created_at": int(time.time()),
        "username": username,
        "name": name,
        "email": email,
        "number": number,
        "password_hash": password_hash
    }

    # Send OTP
    try:
        send_otp_email_resend(email, otp)
    except Exception as e:
        print("OTP email failed:", str(e))
        return f"Failed to send OTP email. Details: {str(e)}", 500

    return render_template("val.html")


# OTP verification route (already POST)
@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    ensure_users_table()

    pending = session.get("pending_signup")
    if not pending:
        return render_template("signup.html")

    message = request.form.get('message', '').strip()
    if not message.isdigit():
        return render_template("val.html")  # invalid OTP format

    # Expiry check
    created_at = int(pending.get("otp_created_at", 0))
    if int(time.time()) - created_at > OTP_TTL_SECONDS:
        session.pop("pending_signup", None)
        return "OTP expired. Please sign up again.", 400

    # Verify OTP
    if int(message) != int(pending["otp"]):
        return render_template("val.html")  # wrong OTP

    # OTP valid -> save user
    con = get_db_connection()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO info (user, email, password, mobile, name) VALUES (?, ?, ?, ?, ?)",
        (pending["username"], pending["email"], pending["password_hash"], pending["number"], pending["name"])
    )
    con.commit()
    con.close()

    # Clear pending signup
    session.pop("pending_signup", None)

    return render_template("signin.html")


# ✅ UPDATED: SIGNIN now POST (no credentials in URL)
@app.route("/signin", methods=["POST"])
def signin():
    ensure_users_table()

    username = request.form.get('user', '').strip()
    password = request.form.get('password', '')

    if not username or not password:
        return render_template("signin.html")

    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT user, password FROM info WHERE user = ?", (username,))
    row = cur.fetchone()
    con.close()

    if not row:
        return render_template("signin.html")

    stored_hash = (row["password"] or "").strip()

    # ✅ bcrypt check
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        # If you had old plaintext passwords before, this prevents crashing.
        # You can remove this fallback once you reset your DB.
        ok = (password == stored_hash)

    if ok:
        return render_template("index.html")

    return render_template("signin.html")


@app.route("/notebook")
def notebook():
    return render_template("Notebook.html")


if __name__ == '__main__':
    app.run(debug=False)
