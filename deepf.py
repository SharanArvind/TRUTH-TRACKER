from deepface import DeepFace
import cv2

def apply_deepfake_to_image(image_path):
    try:
        # Load the image
        frame = cv2.imread(image_path)

        # Use DeepFace for face recognition and manipulation
        result = DeepFace.beautify(frame, model='VGG-Face', filter='smooth')
        beautified_frame = result.get('image')
        cv2.imshow('Deepfake Image', beautified_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", str(e))

# Provide the path to the image
image_path = 'jeff.jpeg'
apply_deepfake_to_image(image_path)