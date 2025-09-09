Iron Lady FAQ Chatbot

A simple FAQ chatbot built with Python + Flask that answers questions about Iron Lady’s leadership programs.
It supports rule-based FAQ matching and an optional AI fallback using the OpenAI API for more natural conversations.

🛠 Tools / Tech Stack

Python 3.10+

Flask (web framework)

HTML, CSS, JavaScript (frontend chat UI)

JSON (FAQ knowledge base)

FuzzyWuzzy + python-Levenshtein (for better matching, optional)

OpenAI API (optional AI fallback)

🚀 Features Implemented

Answer FAQs:

What programs does Iron Lady offer?

What is the program duration?

Is the program online or offline?

Are certificates provided?

Who are the mentors/coaches?

Simple rule-based FAQ matcher (with fuzzy matching).

Web-based chat interface (HTML/JS frontend).

Optional AI fallback (OpenAI GPT) when no FAQ match is found.

Shows answer source (faq | ai | fallback) in chat.

📂 Project Structure
iron-lady-chatbot/
├─ app.py                # Flask backend
├─ requirements.txt      # Dependencies
├─ data/
│  └─ faq.json           # FAQ knowledge base
├─ templates/
│  └─ index.html         # Chat UI
└─ static/               # (optional: custom CSS/JS)

▶️ How to Run
1. Clone / Download
git clone https://github.com/yourusername/iron-lady-chatbot.git
cd iron-lady-chatbot

2. Setup Virtual Environment
python -m venv venv
# Activate it:
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. (Optional) Configure OpenAI API

If you want AI fallback responses:

# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Linux / macOS
export OPENAI_API_KEY="your_api_key_here"

5. Run the App
python app.py


Visit http://127.0.0.1:5000
 in your browser 🚀.