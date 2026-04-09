import pyttsx3

# Initialize engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)   # Speed of speech
engine.setProperty('volume', 1.0) # Volume (0.0 to 1.0)

# Speak text
engine.say("Hello Roshan, this is a text to speech test in Python.")

# Run and wait
engine.runAndWait()