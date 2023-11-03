import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights

# Define a custom model using ResNet-18 as the backbone
class FakeImageDetector(nn.Module):
    def __init__(self):
        super(FakeImageDetector, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify the final classification layer for your specific task
        self.model.fc = nn.Linear(512, 2)
        # Assuming 2 classes for fake and real

    def forward(self, x):
        x = self.model(x)
        return x

# Create an instance of your custom model
model = FakeImageDetector()
model.eval()  # Set the model to evaluation mode


# Define a function for image preprocessing using torchvision.transforms
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App with added features
def main():
    st.title("Fake Image Detector")
    st.sidebar.title("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess and make prediction
        processed_image = preprocess_image(uploaded_file)
        with torch.no_grad():
            prediction = model(processed_image)
        result = torch.argmax(prediction).item()
        if result == 0:
            result_text = "Fake"
        else:
            result_text = "Real"

        # Display confidence score
        confidence_score = torch.softmax(prediction, dim=1)[0][result].item()
        st.write(f"The image is {result_text} with confidence: {confidence_score:.2f}")

        # Add an explanation (for demonstration, using random value)
        explanation = st.button("Show Explanation")
        if explanation:
            st.write("Explanation: This is a random explanation.")

if __name__ == '__main__':
    main()