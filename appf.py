import subprocess
import html
import cv2
import os
import imageio
import numpy as np
from flask import Flask, jsonify, render_template,send_from_directory,send_file, request,session,redirect, url_for
from googleapiclient.discovery import build
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from pytube import YouTube
from keras.preprocessing import image
from classifiers import Meso4
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import newspaper
from transformers import T5ForConditionalGeneration, T5Tokenizer

from pymongo import MongoClient
from finalaudiocode.audiofinal import main


app = Flask(__name__)

app.static_folder = 'templates'
app.secret_key = 'HKGKJWBEIY%#^@VEHJWV'

# Configure MongoDB connection
mongo_uri = "mongodb+srv://webuild:zDEvvvSPzT7ZcUNE@contruction.y5uhaai.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["dface"]
collection = db['fakeimage']


# Route for the home page
@app.route('/')
def home():
    return render_template('front.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Retrieve the user from the database
        users = db.users
        user = users.find_one({'email': email, 'password': password})

        if user:
            session['email'] = user['email']
            session['username'] = user['username']
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return "Login failed"
    return render_template('login.html')


@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        # Get user data from the signup form
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Insert the user into the database
        users = db.users
        users.insert_one({'username': username, 'password': password, 'email': email})

        return render_template('login.html')
    
    return render_template('signup.html')
    

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    user_name = session.get('username')
    if 'logged_in' in session and session['logged_in']:
        return render_template('dashboard.html', user_name=user_name)
    else:
        return "Unauthorized. Please log in first."

@app.context_processor
def inject_username():
    # Define a function to make the username available to all templates
    username = session.get('username', None)
    return dict(user_name=username)


@app.route('/news', methods=['GET','POST'])
def detect_fake_news():
    user_input = ""
    
    if request.method == 'POST':
        
        # Load the pre-trained BERT model and tokenizer for fake news detection
        bert_model_name = "bert-base-uncased"
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Load a pre-trained T5 model and tokenizer for generating explanations
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

        # Define a text classification pipeline
        classifier = pipeline('sentiment-analysis', model=bert_model, tokenizer=bert_tokenizer)

        # Function to analyze fake news
        def analyze_fake_news(news_text):
            try:
                # Classify the news text
                result = classifier(news_text)
                return result
            except Exception as e:
                return str(e)

        # Function to scrape a news article from a URL
        def scrape_article(url):
            article = newspaper.Article(url)
            article.download()
            article.parse()
            return article.text

        # Function to generate explanations using T5
        def generate_explanation(news_text, classification_result):
            input_text = f"Explain why the news article is classified as {classification_result}: {news_text}"

            # Tokenize and generate an explanation
            input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=200, truncation=True)
            generated_ids = t5_model.generate(input_ids, max_length=200, num_return_sequences=1, early_stopping=False)

            explanation = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return explanation

        user_input = request.form['article_url']
        analysis_result = analyze_fake_news(user_input)

        if isinstance(analysis_result, str):
            return render_template('news.html', response=f"Error analyzing fake news: {analysis_result}")
        else:
            label = analysis_result[0]['label']
            confidence = analysis_result[0]['score'] * 100
            confidence_threshold = 70  # Set your desired threshold here
            is_fake = confidence >= confidence_threshold

            labels = ['Real', 'Fake']

            predicted_label = 1 if is_fake else 0

            # Scraping the news article from a URL (you can replace the URL)
            article_url = user_input
            news_text = scrape_article(article_url)

            # Generating an explanation for the classification result
            explanation = generate_explanation(news_text, label)

            return render_template('news.html', news_text=user_input, label=label, confidence=confidence,predicted_label=predicted_label, is_fake=is_fake, explanation=explanation, labels=labels)

    return render_template('news.html')

@app.route('/uploadfakenews', methods=['POST'])
def uploadnews():
        # Get data from the form
        news_text = request.form.get('news_text')
        label = request.form.get('label')
        confidence = request.form.get('confidence')
        predicted_label = request.form.get('predicted_label')
        is_fake = request.form.get('is_fake')
        explanation = request.form.get('explanation')

        document = {
            'news_text': news_text,
            'label': label,
            'confidence': confidence,
            'predicted_label': predicted_label,
            'is_fake': is_fake,
            'explanation': explanation
        }

        # Insert the document into the MongoDB collection
        collection.insert_one(document)

        return render_template('news.html', news_text=news_text, label=label, confidence=confidence, predicted_label=predicted_label, is_fake=is_fake, explanation=explanation)


@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        # Handle the audio file upload (use the 'request.files' object)
        audio_file = request.files['audio_file']

        # Save the uploaded audio file to a temporary location
        temp_path = 'audio.wav'
        audio_file.save(temp_path)

        # Call your audio classification function
        explanation, transcription = main(temp_path)

        # Render the 'audio.html' template and pass the results
        return render_template('audio.html', classification_result=explanation, transcribed_text=transcription)

    return render_template('audio.html', classification_result=None, transcribed_text=None)

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/streamlit')
def streamlit():
    subprocess.Popen(['python', '-m', 'streamlit', 'run', 'image.py'])
    return render_template('image.html')

@app.route('/videos/<path:filename>')
def download_file(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/images/image1')
def get_image1():
    image_directory = 'images'
    return send_file(f'{image_directory}/images.jpeg')

@app.route('/images/image2')
def get_image2():
    image_directory = 'images'
    return send_file(f'{image_directory}/imga.jpg')

@app.route('/images/image3')
def get_image3():
    image_directory = 'images'
    return send_file(f'{image_directory}/audio.jpg')

@app.route('/images/image4')
def get_image4():
    image_directory = 'images'
    return send_file(f'{image_directory}/ani4.png')

@app.route('/images/image5')
def get_image5():
    image_directory = 'images'
    return send_file(f'{image_directory}/realcam.jpg')

@app.route('/report')
def report():
    return render_template('report.html')

# Route for the fake_news_finder page
@app.route('/fake_news_finder', methods=['GET', 'POST'])
def fake_news_finder():
    if request.method == 'POST':
        # Get the keyword and rate limit from the form
        keyword = request.form['keyword']
        rate_limit = int(request.form['rate_limit'])
        
        api_key = "AIzaSyBPHs1Pq49RrKiW1BIFl2uJHYrwa7cpeyY"
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        model_name = "gpt2"
        model = TFAutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Search YouTube for videos with the keyword
        search_response = youtube.search().list(
            part="snippet",
            q=keyword,
            type="video",
            maxResults=rate_limit
        ).execute()

        videos = []
        for item in search_response["items"]:
            video_title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            channel_title = item["snippet"]["channelTitle"]
            published_at = item["snippet"]["publishedAt"]

            # Retrieve video details
            video_response = youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()

            view_count = video_response["items"][0]["statistics"].get("viewCount", 0)
            like_count = video_response["items"][0]["statistics"].get("likeCount", 0)

            videos.append({
                'video_title': video_title,
                'video_id': video_id,
                'channel_title': channel_title,
                'published_at': published_at,
                'view_count': view_count,
                'like_count': like_count
            })

        # Retrieve keyword-related comments
        comments = []
        for video in videos:
            video_id = video['video_id']
            comment_response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                searchTerms=keyword,
                maxResults=rate_limit
            ).execute()

            if "items" in comment_response:
                for comment in comment_response["items"]:
                    comment_text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comment_text = html.unescape(comment_text)
                    comments.append(comment_text)

        prompt = f"Is it true that {keyword}?"
        inputs = tokenizer.encode(prompt, return_tensors="tf", add_special_tokens=True)

        # Generate response from GPT-2 model
        output = model.generate(inputs, max_length=50, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return render_template('fake_news_finder.html', videos=videos, comments=comments, response=response)

    return render_template('fake_news_finder.html', videos=None, comments=None, response=None)


# Route for the fake_video_detector page
@app.route('/fake_video_detector/', methods=['GET', 'POST'])
def fake_video_detector():
    if request.method == 'POST':
        if 'video-file' in request.files:
            video_file = request.files['video-file']
            temp_file_path = 'temp_video.mp4'
            video_file.save(temp_file_path)

            # Process the video
            duration = calculate_video_duration(temp_file_path)
            predictions = process_video(temp_file_path, model)

            if predictions:
                classification, confidence = classify_video(predictions)
                result = {
                    'classification': classification,
                    'confidence': confidence,
                    'duration': duration
                }
                return jsonify(result)  # Return the result as JSON
            else:
                return jsonify({'error': 'Error occurred during video processing.'})

        else:
            return jsonify({'error': 'No video file uploaded.'}), 400

    return render_template('fake_video_detector.html')


# Route for the working page
@app.route('/working/')
def working():
    return render_template('working.html')

# Route for the privacy policy page
@app.route('/privacy_policy/')
def privacy_policy():
    return render_template('privacy_policy.html')


meso4_model = None
vgg16_model = None

def load_meso4_model():
    global meso4_model
    if meso4_model is None:
        meso4_model = Meso4()
        meso4_model.load('./weights/Meso4_DF.h5')
        
def load_vgg16_model():
    global vgg16_model
    if vgg16_model is None:
        vgg16_model = VGG16(weights='imagenet', include_top=False)

        
@app.route('/fake_v/', methods=['GET', 'POST'])
def index():
    cap = None
    if request.method == 'POST':
        video_path = request.form.get('video_input')
        if 'video_input' in request.files:
            file = request.files['video_input']
                        
            def allowed_file(filename):
                ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
                # Add more file extensions as needed
                return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join('uploads', filename))
                video_path = os.path.join('uploads', filename)
                cap = cv2.VideoCapture(video_path)
            else:
                raise ValueError("Invalid file format")
        elif video_path.startswith('https'):
                try:
                    print("In url")
                    yt = YouTube(video_path)
                    ys = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                    if ys:
                        filename = yt.title  # Do not include the file extension here
                        ys.download(output_path='backend', filename=filename + '.mp4')
                        youtube_video = os.path.join('backend', filename + '.mp4')
                        cap = cv2.VideoCapture(youtube_video)
                        print(cap)
                    else:
                        raise ValueError("No suitable video stream available for download")
                except Exception as e:
                    print(f"Error downloading YouTube video: {e}")

        else:
            raise ValueError("Unsupported input video format")
        
        load_vgg16_model()
        load_meso4_model()
        
        
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
            
            
        frame_faces = []

        if cap is not None:
            print("Video capture successful")
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    print("Video capture unsuccessful")
                    break

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face = frame[y:y + h, x:x + w]
                    features = extract_features_from_image(face)
                    best_match_idx = find_best_match(features, known_people_features)
                    matched_name = known_people_names[best_match_idx]

                    frame_faces.append(matched_name)

            cap.release()
            cv2.destroyAllWindows()
            print("Video processing completed")

            if frame_faces:
                most_common_face = Counter(frame_faces).most_common(1)[0]
            recognized_person = most_common_face[0]
        

            return jsonify({
                'recognized_person': recognized_person,
                'video_input': video_path,
            })

    return render_template('fakev.html')

@app.route('/analyze_deepfake', methods=['POST'])
def analyze_deepfake():
    data = request.get_json()
    response = data.get('response')
    video_path = data.get('video_input')

    print("response:", response)
    print("video_path:", video_path)


    if response == "yes":
        recognized_person = data.get('recognized_person')
        print("recognized_person:", recognized_person)
        recognized_person_video_path = os.path.join('real_videos', recognized_person, '1027 (1)(2).mp4')

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
            result = f'The recognized person video is likely real (Meso4 Score: {deepfake_score})'
        else:
            result = f'The recognized person video is likely fake (Meso4 Score: {deepfake_score})'
    else:
        video_path = request.form.get('video_input')
        print("video-path:",video_path)
        print("In response No")
        # Analyze the entire video

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

        entire_video_score = predict_deepfake(video_path)

        if entire_video_score < 0.5:
            result = f'The entire video is likely real (Meso4 Score: {entire_video_score})'
        else:
            result = f'The entire video is likely fake (Meso4 Score: {entire_video_score})'

    return jsonify({"result": result})


def calculate_video_duration(video_path):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Unable to open the video file. {e}")
        return None

    duration = video.get_meta_data()['duration']
    video.close()

    return duration

def process_video(video_path, model):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Unable to open the video file. {e}")
        return []

    frame_count = len(video)
    frame_height, frame_width = video.get_meta_data()['source_size'][:2]

    print(f"Video info: {frame_count} frames, {frame_width}x{frame_height} resolution")

    predictions = []

    try:
        for i, frame in enumerate(video):
            # Resize and preprocess the frame
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess_input(frame)

            # Make a prediction
            prediction = model.predict(frame)
            predictions.append(prediction[0][0])  # Append the prediction result (fake score)

    except Exception as e:
        print(f"Error occurred during video processing. {e}")
        return []

    return predictions

def classify_video(predictions, threshold=0.5):
    if not predictions:
        return 'unknown', 0

    fake_count = sum(1 for score in predictions if score > threshold)
    real_count = len(predictions) - fake_count

    fake_score = fake_count / len(predictions)
    real_score = real_count / len(predictions)

    if fake_score > real_score:
        return 'fake', fake_score
    else:
        return 'real', real_score
    
    
# Define the route to handle form submissions
@app.route('/report', methods=['POST'])
def report_user():
    if request.method == 'POST':
        # Get the submitted report user
        report_user = request.form.get('reportUser')

        # Save the report user to a file (you can use any format you prefer)
        with open('report_users.txt', 'a') as file:
            file.write(report_user + '\n')

        return "Report submitted successfully."
    
@app.route('/realcam', methods=['GET','POST'])
def real_cam():
    if request.method == 'POST':
    
        return render_template('realcam.com')

    return render_template('realcam.html')


if __name__ == '__main__':
    # Load the pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new output layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])

    app.run(debug=True)
