import argparse
import time
import torch
from torchvision import transforms
from PIL import Image
import cv2
from models.model import SmileClassifier, FaceDetector

def main():
    parser = argparse.ArgumentParser(description="realtime smile detection")
    parser.add_argument('--model', type=str, required=True, help='path to the trained model (best_model.pth)')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0 etc.')
    parser.add_argument('--camera', type=int, default=0, help='camera index')
    parser.add_argument('--color', type=tuple, default=(0, 255, 0), help='Tracking box color')
    parser.add_argument('--mirror', action='store_true', help='mirror display')
    parser.add_argument('--interval', type=int, default=1, help='how many frames to wait before doing inference (>=1)')
    parser.add_argument('--save_path', type=str, default='', help='path to save video (empty to not save)')
    args = parser.parse_args()

    device = torch.device(args.device)
    color = args.color

    # Load model
    model = SmileClassifier(pretrained=False).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Face detection
    face_detector = FaceDetector('checkpoints/haarcascade_frontalface_default.xml')

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("cannot open camera")
        return

    writer = None
    if args.save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(args.save_path, fourcc, fps, (w, h))

    last_predictions = {}
    frame_idx = 0
    t_last = time.time()
    fps_smoothed = 0.0

    print("press q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, exiting")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        gray_faces = face_detector.detect_face(frame)

        current_preds = {}
        do_infer = (frame_idx % max(1, args.interval) == 0)

        for (x, y, w, h) in gray_faces:
            face_img_bgr = frame[y:y + h, x:x + w]
            pred_label = '...'
            score = 0.0
            if do_infer:
                face_pil = Image.fromarray(cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB))
                with torch.no_grad():
                    tensor = data_transforms(face_pil).unsqueeze(0).to(device)
                    out = model(tensor)
                    score = out.item()
                    pred_label = 'smile' if score > 0.5 else 'non_smile'
                current_preds[(x, y, w, h)] = (pred_label, score)
            else:
                if last_predictions:
                    pred_label, score = list(last_predictions.values())[-1]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame,
                        f"{pred_label} {score:.2f}",
                        (x, y - 10 if y - 10 > 20 else y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2)

        if do_infer and current_preds:
            last_predictions = current_preds

        # FPS
        now = time.time()
        dt = now - t_last
        t_last = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smoothed = (fps_smoothed * 0.9 + inst_fps * 0.1) if fps_smoothed > 0 else inst_fps
        cv2.putText(frame, f"FPS: {fps_smoothed:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow("Smile Recognition", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"video saved to {args.save_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()