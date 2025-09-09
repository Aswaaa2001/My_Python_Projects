from flask import Flask, request, jsonify, render_template
import os, json, difflib

# Optional: comment out if you don't want AI fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

app = Flask(__name__)

# Load FAQ data
with open("data/faq.json", "r", encoding="utf-8") as f:
    FAQ = json.load(f)

FAQ_QUESTIONS = [entry["question"] for entry in FAQ]

def find_best_faq(user_msg):
    # 1) Try fuzzy ratio over entire question text
    lower_msg = user_msg.lower()
    best_q = None
    best_score = 0.0
    for q in FAQ_QUESTIONS:
        score = difflib.SequenceMatcher(None, lower_msg, q.lower()).ratio()
        if score > best_score:
            best_score = score
            best_q = q
    # 2) If score is high enough, return the FAQ answer
    if best_score >= 0.55:
        for entry in FAQ:
            if entry["question"] == best_q:
                return entry["answer"], best_score
    # 3) Keyword fallback
    for entry in FAQ:
        for kw in (entry.get("keywords") or []):
            if kw.lower() in lower_msg:
                return entry["answer"], 0.45
    return None, best_score

@app.route("/")
def index():
    return render_template("intex.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    if not user_msg:
        return jsonify({"reply": "Please send a question."})

    # Try FAQ match
    answer, score = find_best_faq(user_msg)
    if answer:
        return jsonify({"reply": answer, "source": "faq", "score": score})

    # Fallback to AI if OpenAI is configured and available
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and OPENAI_AVAILABLE:
        openai.api_key = api_key
        system_prompt = (
            "You are a concise assistant that only answers FAQs about 'Iron Lady' leadership programs. "
            "If the question is outside that scope, politely say you don't know and direct to contact support@example.com."
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=200,
                temperature=0.2
            )
            ai_reply = resp["choices"][0]["message"]["content"].strip()
            return jsonify({"reply": ai_reply, "source": "ai"})
        except Exception as e:
            return jsonify({"reply": f"Sorry, error contacting AI: {e}"}), 500

    # Final fallback
    return jsonify({
        "reply": "I don't have a confident answer for that. You can ask about programs, duration, online/offline, certificates, or mentors.",
        "source": "fallback"
    })

if __name__ == "__main__":
    app.run(debug=True)
