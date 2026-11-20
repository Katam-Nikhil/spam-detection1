"""Flask website for classifying a message as spam or not spam."""
from pathlib import Path
import joblib
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "spam_model.pkl"

app = Flask(__name__)
model = None

def load_model():
    global model
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run 'python model.py' first.")
    model = joblib.load(MODEL_FILE)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            result = {"label": "Please enter a message.", "kind": "warning"}
        else:
            prediction = model.predict([message])[0]
            if prediction == "spam":
                result = {"label": "Spam message detected", "kind": "spam"}
            else:
                result = {"label": "This message looks like ham (not spam)", "kind": "ham"}
    return render_template("index.html", result=result, message=message)

if __name__ == "__main__":
    load_model()
    app.run(debug=True)
