import torch
import torch.nn as nn
import torchvision.models as models
import cv2

class SmileClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(SmileClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.resnet.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

class FaceDetector():
    def __init__(self, xml_path):
        self.face_cascade = cv2.CascadeClassifier(xml_path)

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

if __name__ == '__main__':
    model = SmileClassifier(pretrained=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    output = model(input)

    print("Output shape:", output.shape)
    print("Output value:", output.item())
