import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from pytube import YouTube
from keras.preprocessing import image
from classifiers import Meso4

meso4_model = Meso4()

# Load the VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False)

meso4_model.load('./weights/Meso4_DF.h5')


def enhance_image(image, target_size=(224, 224)):
    enhanced_image = cv2.resize(image, target_size)
    return enhanced_image


def extract_features_from_image(img):
    img = enhance_image(img)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg16_model.predict(img)
    return features


def extract_features(img_path):
    try:
        img = cv2.imread(img_path)
        if img is not None:
            img = extract_features_from_image(img)
            return img
        else:
            return None
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return None


def find_best_match(query_features, known_people_features):
    if query_features is None:
        return -1

    similarities = []
    query_features = query_features.flatten()

    for known_features in known_people_features:
        if known_features is not None:
            known_features = known_features.flatten()
            similarities.append(cosine_similarity([query_features], [known_features])[0][0])
        else:
            similarities.append(0)

    best_match_idx = np.argmax(similarities)
    return best_match_idx


known_people_features = []
known_people_names = []

for person_name in os.listdir('real_videos'):
    person_dir = os.path.join('real_videos', person_name)
    entries = os.listdir(person_dir)
    paths = [os.path.join(person_dir, entry) for entry in entries if entry.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]

    for path in paths:
        features = extract_features(path)
        if features is not None:
            known_people_features.append(features)
            known_people_names.append(person_name)

# Load the input video (local .mp4 or YouTube URL)
input_video = '1027 (1)(1).mp4'
#https://youtu.be/XuKUkyPegBE?si=JRFGIJ9wzokBAEct

if input_video.endswith('.mp4'):
    cap = cv2.VideoCapture(input_video)
elif input_video.startswith('https'):
    try:
        yt = YouTube(input_video)
        ys = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if ys:
            filename = yt.title  # Do not include the file extension here
            ys.download(output_path='backend', filename=filename+'.mp4')
            youtube_video=os.path.join('backend', filename + '.mp4')
            cap = cv2.VideoCapture(youtube_video)
        else:
            raise ValueError("No suitable video stream available for download")
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        cap = None
else:
    raise ValueError("Unsupported input video format")

frame_faces = []

if cap is not None:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            features = extract_features_from_image(face)
            best_match_idx = find_best_match(features, known_people_features)
            matched_name = known_people_names[best_match_idx]

            frame_faces.append(matched_name)

    cap.release()
    cv2.destroyAllWindows()

if frame_faces:
    most_common_face = Counter(frame_faces).most_common(1)[0]
    print(f'Most common recognized person in the video: {most_common_face[0]}')
    recognized_person = most_common_face[0]
    recognized_person_video_path = os.path.join('real_videos',
                                                recognized_person, 'video.mp4')

    def predict_deepfake(video_path):
        video_frames = []

        # Capture frames from the video
        video_capture = cv2.VideoCapture(video_path)
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            video_frames.append(frame)

        # Perform Meso4 predictions
        predictions = []
        for frame in video_frames:
            # Preprocess the frame
            frame = cv2.resize(frame, (256, 256))
            frame = image.img_to_array(frame)
            frame = np.expand_dims(frame, axis=0)
            frame = frame / 255.0  # Normalize

            # Make predictions
            prediction = meso4_model.predict(frame)
            predictions.append(prediction)

        # Calculate an average prediction score
        average_score = np.mean(predictions)

        return average_score

    # Analyze if the recognized person's video is real or fake
    deepfake_score = predict_deepfake(recognized_person_video_path)

    if deepfake_score < 0.5:
        print(f'The recognized person video is likely real (Meso4 Score: {deepfake_score})')
    else:
        print(f'The recognized person video is likely fake (Meso4 Score: {deepfake_score})')
else:
    
    def predict_deepfake(video_path):
        video_frames = []

        # Capture frames from the video
        video_capture = cv2.VideoCapture(video_path)
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            video_frames.append(frame)

        # Perform Meso4 predictions
        predictions = []
        for frame in video_frames:
            # Preprocess the frame
            frame = cv2.resize(frame, (256, 256))
            frame = image.img_to_array(frame)
            frame = np.expand_dims(frame, axis=0)
            frame = frame / 255.0  # Normalize

            # Make predictions
            prediction = meso4_model.predict(frame)
            predictions.append(prediction)

        # Calculate an average prediction score
        average_score = np.mean(predictions)

        return average_score
    
    entire_video_score = predict_deepfake(input_video)

    if entire_video_score < 0.5:
        print(f'The entire video is likely real (Meso4 Score: {entire_video_score})')
    else:
        print(f'The entire video is likely fake (Meso4 Score: {entire_video_score}')
