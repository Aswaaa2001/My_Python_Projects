import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser

engine = pyttsx3.init()

def speak(audio):
    """Convert text to speech."""
    engine.say(audio)
    engine.runAndWait()

def get_time():
    """Get the current time and speak it."""
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    speak("The current time is " + current_time)

def get_date():
    """Get the current date and speak it."""
    year = datetime.datetime.now().strftime("%Y")
    month = datetime.datetime.now().strftime("%m")
    day = datetime.datetime.now().strftime("%d")
    speak("The current date is " + month + "/" + day + "/" + year)

def wish_me():
    """Greet the user based on the current time."""
    hour = datetime.datetime.now().hour
    if hour >= 3 and hour < 12:
        speak("Good morning")
    elif hour >= 12 and hour < 16:
        speak("Good afternoon")
    elif hour >= 16 and hour < 19:
        speak("Good evening")
    else:
        speak("Good night")

    speak("I'm Thakkudu, an assistant created by Ashwin.")

def take_command():
    """Listen for a command from the user and return it."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 0.5
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print("You said: " + query)
        return query.lower()  

    except Exception as e:
        print("Error recognizing speech:", str(e))
        speak("Sorry, I didn't catch that. Please say it again.")
        return None

def main():
    """Main function to run the assistant."""
    wish_me()

    while True:
        query = take_command()

        if query is None:
            continue  

        query = query.lower()

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
            except Exception as e:
                print("Could not find any results on Wikipedia.")
                speak("Sorry, I could not find any results on Wikipedia.")
        elif 'search in chrome' in query:
            query = query.replace("search in chrome", "")
            speak("Searching in Chrome")
            webbrowser.open("https://www.google.com/search?q=" + query)

        elif 'open youtube' in query:
            speak("Opening YouTube")
            webbrowser.open("https://www.youtube.com")

        elif 'gpt' in query:
            speak("Opening ChatGPT")
            webbrowser.open("https://chat.openai.com")

        elif 'open google' in query:
            speak("Opening Google")
            webbrowser.open("https://www.google.com")
        
        elif 'stop' in query:
            speak("Goodbye Ashwin!")
            break  

        else:
            speak("Sorry, I didn't understand that. Please try again.")

if __name__ == '__main__':
    main()
