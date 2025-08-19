import argparse
import json
import os
import torch
from torchvision import transforms
from PIL import Image
from models.model import SmileClassifier


def predict(image_path, model_path, device):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = SmileClassifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        score = output.item()

        prediction = 'smile' if score > 0.5 else 'non_smile'

    return prediction, score

def main():
    parser = argparse.ArgumentParser(description='Smile Recognition Testing')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda:0 or cpu)')
    args = parser.parse_args()

    prediction, score = predict(args.image, args.model, torch.device(args.device))

    print(f'predict result: {prediction}')
    print(f'smiling score: {score:.4f}')


if __name__ == '__main__':
    main()