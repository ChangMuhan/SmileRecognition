import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.model import SmileClassifier
from dataloader import SmileDataset
from tqdm import tqdm


def train(config):
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        x: SmileDataset(root_dir=os.path.join(config['data_root'], x), transform=data_transforms[x])
        for x in ['train', 'test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=config['batch_size'], shuffle=True, num_workers=4)
        for x in ['train', 'test']
    }

    model = SmileClassifier(pretrained=True).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    num_epochs = config['num_epochs']
    output_root = config['output_root']
    os.makedirs(output_root, exist_ok=True)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).int()
                running_corrects += torch.sum(preds == labels.unsqueeze(1).int())

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_path = os.path.join(output_root, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f'save the best model to: {best_model_path}')

        print()

    print("finished training!")


def main():
    parser = argparse.ArgumentParser(description='Smile Recognition Training')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config)


if __name__ == '__main__':
    main()