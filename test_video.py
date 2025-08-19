import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
from models.model import SmileClassifier, FaceDetector

def predict(image, model, data_transforms, device):
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        score = output.item()
        prediction = 'smile' if score > 0.5 else 'non_smile'
    return prediction, score

def main():
    parser = argparse.ArgumentParser(description='Smile Recognition Testing in Real Time')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--color', type=tuple, default=(0, 255, 0), help='Tracking box color')
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--output_video', type=str, default='output.mp4', help='Path to the output video')
    args = parser.parse_args()

    device = torch.device(args.device)

    model = SmileClassifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    face_detector = FaceDetector('checkpoints/haarcascade_frontalface_default.xml')

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        video_source = int(args.input_video)
    except ValueError:
        video_source = args.input_video
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open video source")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{args.output_video}', fourcc, frame_rate, output_size)

    i = 0
    last_prediction = {}
    print("processing video...")
    while (True):
        # print(i)
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect_face(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            # if i % 25 == 0:
            face_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            prediction, score = predict(face_img, model, data_transforms, device)
            last_prediction[(x,y,w,h)] = (prediction, score)
            # print(last_prediction)
            prediction, score = list(iter(last_prediction.values()))[-1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), args.color, 2)
            cv2.putText(frame, f"{prediction} {score:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, args.color,2)
        out.write(frame)
        i += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"the output video is saved as: {args.output_video}")

if __name__ == "__main__":
    main()