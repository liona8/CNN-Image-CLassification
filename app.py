import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchsummary import summary
# import cnnproject_weatherclassification
from PIL import Image

image_size = 256

tf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
    ])

convnext_tiny = models.convnext_tiny(weights='IMAGENET1K_V1')

summary(convnext_tiny, (3, 224, 224))

# close all the parameter to be trained
for p in convnext_tiny.parameters():
  p.requires_grad = False

for n,p in convnext_tiny.named_parameters():
  # print(n)
  if 'features.1' in n or 'features.4' in n :
    p.requires_grad = True

# convnext_tiny.features[-1].shape()

summary(convnext_tiny, (3, 224, 224))

num_class = 11

class CNNModel(nn.Module):

  def __init__(self, image_depth):

    super().__init__()
    self.convnext_tiny = convnext_tiny

    self.fc1 = nn.Linear(in_features=1000,out_features=512)
    self.fc2 = nn.Linear(in_features=512,out_features=128)
    self.fc3 = nn.Linear(in_features=128,out_features=32)
    self.out = nn.Linear(in_features=32,out_features=num_class)

  def forward(self, x):
    x = self.convnext_tiny(x)

    #fully-connected classifiers.
    x = torch.flatten(x, start_dim=1) #we need to all the dimensions except the batch. 0 is batch
    x = F.relu(self.fc1(x))
    x = F.dropout(x,p=0.45)
    x = F.relu(self.fc2(x))
    x = F.dropout(x,p=0.25)
    x = F.relu(self.fc3(x))
    x = F.dropout(x,p=0.1)
    x = self.out(x)

    return x

model = CNNModel(3)

model = torch.load('convnextCNN_model.pth', map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

class_names = ['dew','fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Function to preprocess and classify the uploaded image
def classify_image(image):
    # Apply necessary image transformations
    img = tf(image).unsqueeze(0)  # Add batch dimension

    # Classify the image
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Compute class probabilities
        confidence, predicted = torch.max(probabilities, 1)          # Get predicted class and confidence

    return predicted.item(), confidence.item()  # Return both class index and confidence score

# Streamlit UI
st.title("Weather Classification App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify the image
    label, confidence = classify_image(image)

    # Display the prediction and confidence
    st.write(f"Prediction: {class_names[label]} with confidence {confidence:.2f}")