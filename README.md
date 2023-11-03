*TRUTH TRACKER:*

  Truth Tracker is the process of identifying manipulated or synthetic media content, such as *videos, **images, or **audio*, which has been artificially created using deep learning techniques. 
Detecting deepfakes is crucial for maintaining the integrity of digital content and preventing the spread of misleading or fraudulent information. Various methods are employed for deep fake detection,
including forensic analysis, anomaly detection, and the use of machine learning algorithms to recognize subtle inconsistencies and artifacts in manipulated media.

*FEATURES:*

  -*Fake News Detection:*  Utilizes BERT and T5 models to classify news articles as real or fake based on user-provided URLs.

  -*Audio Analysis:* Allows users to upload audio files for processing, which involves tasks like speech recognition and feature extraction.

  -*Image Processing:* Enables users to upload images for various tasks like face recognition and deepfake detection using OpenCV.

  -*Video Analysis:* Users can upload videos to analyze for deepfake content, processed frame by frame using the Meso4 model.

  -*YouTube Data Retrieval:* Fetches information from YouTube based on user-provided keywords using the YouTube Data API.

  -*Real-time Webcam Access:* Provides users with access to their webcam for live video analysis, potentially for tasks like face recognition.

*INSTALLATION:*

Before you begin, make sure you have Python installed. Then, follow these steps to set up Truth Tracker:

 1. Clone the repository:

   `https://github.com/Vijay-0307/backend.git
    cd backend`

 2. Install the requirements:

    `pip install requirements.txt`
  
4. To run the code:

    ```python appf.py```

*WORKING:*

1. *DEEPFAKE VIDEO*

![vedio](https://github.com/Vijay-0307/backend/assets/95033427/56e81cb7-71d5-483c-8a9a-c99fddfe1b09)

- Content: Users can upload a video for analysis. The application uses a pre-trained Meso4 model to predict whether the video contains deepfake content.
- Packages Used: imageio, OpenCV (cv2), numpy
- How It Works: 
    - When a user uploads a video, the application captures frames from the video using the imageio package.
    - Each frame is then processed using OpenCV to potentially identify deepfake content, using the pre-trained Meso4 model.
    - The application aggregates the predictions from all frames to provide an analysis result for the entire video.
    - The results are then presented to the user.

2. *DEEPFAKE AUDIO*

![audio](https://github.com/Vijay-0307/backend/assets/95033427/9f22cf54-4d7e-4f7f-83dc-bb1909e59dd9)

   - Content: Users can upload an audio file for analysis. The file is processed to extract relevant information.
- Packages Used: imageio, finalaudiocode (assuming it contains the audio analysis functions)
- How It Works: 
    - When a user uploads an audio file, the application saves it to a temporary location on the server.
    - The audio file is then passed through the finalaudiocode package (assuming it contains the necessary audio analysis functions) for processing.
    - This may involve tasks like speech recognition, feature extraction, or any specific analysis relevant to the application's purpose.
    - The results of the analysis are then presented to the user.

3. *DEEPFAKE NEWS*

![news](https://github.com/Vijay-0307/backend/assets/95033427/9e3cced2-bc76-489d-81af-657ade45e339)

   - Content: Users can input a news article URL, which is then processed to detect fake news using a pre-trained BERT model.
- Packages Used: newspaper, transformers, collections
- How It Works: 
    - When a user inputs a news article URL, the application uses the newspaper package to download and extract the text content from the article.
    - The extracted text is then passed through a pre-trained BERT model for fake news classification. BERT is a powerful transformer-based model that excels in natural language processing tasks.
    - The BERT model outputs a probability score indicating the likelihood of the news being fake.
    - Additionally, the application uses a T5 model to generate an explanation for the classification result. The T5 model is fine-tuned to generate explanations based on the news content and classification label.

4. *DEEPFAKE IMAGE*

  ![image](https://github.com/Vijay-0307/backend/assets/95033427/8ede1b66-3dfb-4fbb-8b76-90186463eb73)

   - Content: Users can perform various tasks related to image processing, including face recognition and deepfake detection.
- Packages Used: OpenCV (cv2), scikit-learn
- How It Works: 
    - Users can upload images, which are then processed using OpenCV and potentially other libraries like scikit-learn for specific tasks.
    - For face recognition, the application may use pre-trained models or algorithms to detect and recognize faces in the uploaded images.
    - For deepfake detection, the application may use a pre-trained model like Meso4 to predict whether an image contains deepfake content.
    - The results of these image processing tasks are then displayed to the user.

5. *YOUTUBE RETREIVAL*

![youtube](https://github.com/Vijay-0307/backend/assets/95033427/28c96b84-489c-488f-ac3d-b632c5f974bf)

- Content: The application can fetch information from YouTube based on user-provided keywords, using the YouTube Data API.
- Packages Used: googleapiclient (assuming it's used for YouTube API), others related to the specific task
- How It Works: 
    - Users provide keywords to search on YouTube.
    - The application interacts with the YouTube Data API using the googleapiclient package to perform the search.
    - The API returns relevant information about videos, such as titles, view counts, and channel information.
    - The application may apply further filtering or processing to this data before presenting it to the user.

6. *REAL TIME DETECTING*

![realcam](https://github.com/Vijay-0307/backend/assets/95033427/4f6a266f-c78f-4ad8-bc14-7f57f8dab9e6)

- Content: Users can access their webcam in real-time, potentially for live video analysis.
- Packages Used: OpenCV (cv2)
- How It Works: 
    - The application uses OpenCV to access the user's webcam and continuously capture frames in real-time.
    - These frames can be processed for various purposes, such as face recognition or any other live video analysis tasks.
    - The results of the analysis can be displayed in real-time to the user.
