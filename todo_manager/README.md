# 📝 To-Do Manager

A simple **CRUD-based To-Do Manager** built with **Flask + SQLite**.  
It allows users to **add, view, edit, and delete tasks** with a clean and minimal web interface.

---

## 📊 Badges
![Python](https://img.shields.io/badge/python-3.12-blue)
![Flask](https://img.shields.io/badge/flask-2.x-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Status](https://img.shields.io/badge/status-active-success)

---

## 📚 Table of Contents
- [Features](#-features)
- [Installation / Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features
- ✅ Add new tasks  
- ✅ View all tasks  
- ✅ Edit existing tasks  
- ✅ Delete tasks  
- ✅ Mark tasks as done/not done  
- ⚡ Lightweight, beginner-friendly Flask project  

**Problem it solves**: Helps organize daily tasks in a simple and efficient way.

---

## ⚙️ Installation / Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/todo-manager.git
cd todo-manager

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the app
python app.py


App will run on 👉 http://127.0.0.1:5000

🚀 Usage
Add a Task

Fill in the title and description → click Submit

Edit / Update

Click Edit button next to a task → modify → save changes

Delete

Click Delete button next to a task

Example Screenshot

📂 Project Structure
todo_manager/
│── app.py                # Main Flask app
│── todo.db               # SQLite database (auto-created)
│── requirements.txt      # Dependencies
└── templates/            # HTML templates (Jinja2)
    ├── index.html
    ├── add.html
    └── edit.html


⚙️ Configuration

No extra config required.

Database: SQLite auto-created as todo.db.

If using .env, you can set:

FLASK_ENV=development
FLASK_APP=app.py

🤝 Contributing

Fork the repo

Create a new branch (feature-xyz)

Commit your changes (git commit -m "Add new feature")

Push to your branch (git push origin feature-xyz)

Open a Pull Request

📜 License

This project is licensed under the MIT License – free to use, modify, and distribute.


