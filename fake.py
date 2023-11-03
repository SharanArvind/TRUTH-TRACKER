from transformers import FeatureExtractor
import requests

# Initialize the feature extractor
feature_extractor = FeatureExtractor.from_pretrained("facebook/detectron2-resnet50")


# Function to fetch video from URL
def get_video_from_url(url):
    response = requests.get(url)
    with open("1026.mp4", "wb") as video_file:
        video_file.write(response.content)
    return "1026.mp4"


# Prompt user for input (URL of a video)
video_url = input("Enter the URL of a video: ")

# Get video content
try:
    video_path = get_video_from_url(video_url)
except Exception as e:
    print(f"Error fetching video from URL: {e}")
    exit()

# Process the video
inputs = feature_extractor(video_path)

# Processed video features are now available in inputs
# You can perform various tasks based on the extracted features

# Generate report
report = ["Video analysis results:"]  # Add analysis results here

# Create a report file
report_filename = 'video_analysis_report.txt'
with open(report_filename, 'w') as report_file:
    report_file.write('\n'.join(report))

# Print report
print("\nVideo Analysis Report generated successfully.")
print(f"The report has been saved as '{report_filename}'.")