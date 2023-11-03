import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import librosa
import os
from pydub import AudioSegment
import speech_recognition as sr
import pandas as pd

# Define the paths for REAL and FAKE folders
REAL_FOLDER = 'REAL'
FAKE_FOLDER = 'FAKE'

# Function to extract audio features (same as before)
def extract_features(audio_file):
    if audio_file.endswith(".mp3"):
        audio_file = AudioSegment.from_mp3(audio_file)
        audio_file.export("temp.wav", format="wav")
        y, sr = librosa.load("temp.wav")
        os.remove("temp.wav")
    else:
        y, sr = librosa.load(audio_file)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Train the model
def train_model():
    real_files = [os.path.join(REAL_FOLDER, file) for file in os.listdir(REAL_FOLDER)]
    fake_files = [os.path.join(FAKE_FOLDER, file) for file in os.listdir(FAKE_FOLDER)]

    real_features = [extract_features(audio_file) for audio_file in real_files]
    fake_features = [extract_features(audio_file) for audio_file in fake_files]

    X = real_features + fake_features
    y = ['real'] * len(real_features) + ['fake'] * len(fake_features)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)

    return classifier

# Save the model
def save_model(classifier, filename):
    joblib.dump(classifier, filename)

# Load the model
def load_model(filename):
    return joblib.load(filename)

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    try:
        # Check the file extension and convert if necessary
        if audio_file.endswith(".mp3"):
            audio_file = AudioSegment.from_mp3(audio_file)
            audio_file.export("temp.wav", format="wav")
            audio_file = "temp.wav"

        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; check your network connection: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    finally:
        if audio_file == "temp.wav":
            os.remove("temp.wav")

def get_explanation(classification_result):
    if classification_result == 'real':
        return "The audio is classified as real."
    else:
        return "The audio is classified as fake. This classification is based on the extracted features and may not always be accurate. It's important to note that audio classification can sometimes be challenging due to variations in audio quality and other factors"

def main(audio_file_path):
    # Feature extraction (same as before)
    features = extract_features(audio_file_path)

    # Load or train the model
    if os.path.exists('audio_detection_model.joblib'):
        loaded_classifier = load_model('audio_detection_model.joblib')
    else:
        loaded_classifier = train_model()
        save_model(loaded_classifier, 'audio_detection_model.joblib')

    # Transcribe audio
    transcription = transcribe_audio(audio_file_path)

    # Print transcribed text
    print(f"Transcribed Text: {transcription}")

    # Make a prediction
    prediction = loaded_classifier.predict([features])

    # Display result and explanation
    explanation = get_explanation(prediction[0])
    print("Result:")
    print(explanation)
    
    return explanation, transcription

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Classification Script")
    parser.add_argument("audio_file_path", type=str, help="Path to the audio file (e.g., wav or mp3)")
    args = parser.parse_args()
    main(args.audio_file_path)
