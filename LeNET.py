import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


import pathlib

base_dir = pathlib.Path(r"A:\DESKTOP\DL\cats_and_dogs_filtered")
transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

train_dataset=datasets.ImageFolder(
    base_dir/'train',
    transform=transform
)

test_dataset=datasets.ImageFolder(
    base_dir/'validation',
    transform=transform
)

train_loader=DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader=DataLoader(
    test_dataset,
    batch_size=32,   
    shuffle=False
)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2),
          
            nn.Conv2d(6,16,kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2),
          
            nn.Conv2d(16,120,kernel_size=5),
            nn.Tanh()
        )

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(120,84),
            nn.Linear(84,2), 
        )
    def forward(self,x):
        x=self.conv(x)
        x=self.fc(x)
        return x

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=LeNet().to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

epochs=50

for epoch in range(epochs):
    model.train()

    running_loss=0
    score=0
    total=0

    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        
        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()
        

        _,predicted=torch.max(outputs,1)

        total+=labels.size(0)
        score+=(predicted==labels).sum().item()

    train_accuracy=100*score/total

    model.eval()

    val_score=0
    val_total=0
    
    with torch.no_grad():

        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)

            _,predicted=torch.max(outputs,1)

            val_total+=labels.size(0)
            val_score+=(predicted==labels).sum().item()

        val_accuracy=100* val_score/val_total



    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )