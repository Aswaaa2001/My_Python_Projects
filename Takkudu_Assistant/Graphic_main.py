import customtkinter as ctk
from PIL import Image, ImageTk, ImageSequence
import threading
import datetime
import speech_recognition as sr
import pyttsx3
import wikipedia
import webbrowser

# Text-to-speech engine
engine = pyttsx3.init()

# App init
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("Thakkudu Voice Assistant")
app.geometry("600x500")
app.resizable(False, False)

# Title label
title_label = ctk.CTkLabel(app, text="Takkudu - Your Voice Assistant", font=("Arial", 24))
title_label.pack(pady=20)

# GIF area
gif_label = ctk.CTkLabel(app, text="")
gif_label.pack(pady=10)
gif_image = Image.open("circle.gif")
frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA")) for frame in ImageSequence.Iterator(gif_image)]

# Status label
status_label = ctk.CTkLabel(app, text="Status: Idle", font=("Arial", 16))
status_label.pack(pady=10)

# Function to update status
def update_status(text, timeout=2000):
    status_label.configure(text=text)
    if timeout:
        app.after(timeout, lambda: status_label.configure(text="Status: Idle"))

# Animate GIF
def animate(index=0):
    frame = frames[index]
    gif_label.configure(image=frame)
    index = (index + 1) % len(frames)
    app.after(50, animate, index)
animate()

# Voice functions
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def get_time():
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    speak("The current time is " + current_time)

def get_date():
    date = datetime.datetime.now().strftime("%m/%d/%Y")
    speak("The current date is " + date)

def wish_me():
    hour = datetime.datetime.now().hour
    if 3 <= hour < 12:
        speak("Good morning")
    elif 12 <= hour < 16:
        speak("Good afternoon")
    elif 16 <= hour < 19:
        speak("Good evening")
    else:
        speak("Good night")
    speak("I'm Thakkudu, an assistant created by Aswin.")

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        update_status("Listening...")
        r.pause_threshold = 0.5
        audio = r.listen(source)

    try:
        update_status("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print("You said:", query)
        update_status(f"You said: {query}", timeout=2000)
        return query.lower()
    except Exception as e:
        print("Error:", e)
        speak("Sorry, I didn't catch that.")
        update_status("Didn't catch that!", timeout=2000)
        return None

# Main assistant logic
def assistant_main():
    wish_me()
    while True:
        query = take_command()
        if query is None:
            continue
        if 'time' in query:
            get_time()
        elif 'date' in query:
            get_date()
        elif 'wikipedia' in query:
            speak("Searching Wikipedia...")
            query = query.replace("wikipedia", "")
            try:
                result = wikipedia.summary(query, sentences=2)
                print(result)
                speak(result)
            except:
                speak("Sorry, I couldn't find anything.")
        elif 'search in chrome' in query:
            query = query.replace("search in chrome", "")
            speak("Searching in Chrome")
            webbrowser.open(f"https://www.google.com/search?q={query}")
        elif 'open youtube' in query:
            speak("Opening YouTube")
            webbrowser.open("https://www.youtube.com")
        elif 'open google' in query:
            speak("Opening Google")
            webbrowser.open("https://www.google.com")
        elif 'gpt' in query:
            speak("Opening ChatGPT")
            webbrowser.open("https://chat.openai.com")
        elif 'stop' in query or 'exit' in query:
            speak("Goodbye Aswin!")
            update_status("Stopped. Bye 👋")
            break
        else:
            speak("Sorry, I didn't understand that.")

# Thread to run assistant
def run_assistant_thread():
    thread = threading.Thread(target=assistant_main)
    thread.start()

# Start button
start_btn = ctk.CTkButton(app, text="Start Listening", command=run_assistant_thread, font=("Arial", 18))
start_btn.pack(pady=30)

# Footer
footer = ctk.CTkLabel(app, text="Created by Aswin Chandran", font=("Arial", 12))
footer.pack(side="bottom", pady=10)

# Run the app
app.mainloop()
