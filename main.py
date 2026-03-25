from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torchvision import transforms




train_transform = transforms.Compose([
    transforms.Resize((256, 256)),        # resize image
    transforms.RandomResizedCrop(224),    # random crop (better learning)
    transforms.RandomHorizontalFlip(),    
    transforms.ToTensor(),                
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],       # ImageNet mean
        std=[0.229, 0.224, 0.225]         # ImageNet std
    )
])



from torchvision import datasets

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,  
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



alexnet = models.alexnet(pretrained=True)

# Change last layer (IMPORTANT)
alexnet.classifier[6] = nn.Linear(4096, 2)  # example: 2 classes (cat, dog)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.001)

alexnet.train()

for epoch in range(3):  # number of epochs
    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = alexnet(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")



torch.save(alexnet.state_dict(), "alexnet_finetuned.pth")