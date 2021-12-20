import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
from torchvision import transforms, datasets
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from random_erase import RandomErasing_

# hyper-parameter
BATCH_SIZE = 16
EPOCH = 10
LR = 1e-3

# Dataset from torchvision
transform = transforms.Compose([
                                transforms.Resize(512),
                                transforms.CenterCrop(512),
                                transforms.ToTensor()])
data_set = datasets.ImageFolder("dataset/PetImages", transform=transform)
train_set, valid_set, test_set = random_split(data_set, [20000, 2498, 2500],
                                              generator=torch.Generator().manual_seed(87))

transform = transforms.Compose([
                                transforms.Resize(512),
                                transforms.CenterCrop(512),
                                transforms.ToTensor(),
                                RandomErasing_(p=1, value="random")])
augmente_set = datasets.ImageFolder("dataset/PetImages", transform=transform)
train_augmente_set, _, _ = random_split(data_set, [20000, 2498, 2500],
                                        generator=torch.Generator().manual_seed(87))

train_set = train_set + train_augmente_set

# data loader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE * 2)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2)

print(len(train_set), len(valid_set), len(test_set))

# model from torchvision
class ResNet_w_cls(nn.Module):
    def __init__(self):
        super(ResNet_w_cls, self).__init__()
        self.m = nn.Sequential(
            resnet50(),
            nn.Linear(1000, 2)
        )

    def forward(self, x):
        return self.m(x)

model = ResNet_w_cls()
# print(model)

# optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
loss_func = nn.CrossEntropyLoss()
device = torch.device(f"cuda:{sys.argv[2]}" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(device)

writer = SummaryWriter("runs/augment_1")

# main loop
train_epoch_loss = []
valid_epoch_loss = []
test_epoch_loss = []
valid_epoch_acc = []
epoch_acc = []

TRAIN_FROM = 0
# model.load_state_dict(torch.load(f"model/e_{TRAIN_FROM}.pt"))

for epoch in trange(TRAIN_FROM, EPOCH):
    # train
    model.train()
    train_running_loss = 0.0
    for data in tqdm(train_loader):
        imgs, labels = [t.to(device) for t in data]

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

    train_running_loss /= len(train_loader.dataset)
    print(f"epoch: {epoch + 1} train Loss: {train_running_loss}")
    train_epoch_loss += [train_running_loss]
    torch.save(model.state_dict(), f"model/aug/e_{epoch + 1}.pt")

    # valid
    model.eval()
    valid_running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data in tqdm(valid_loader):
            imgs, labels = [t.to(device) for t in data]

            outputs = model(imgs)
            loss = loss_func(outputs, labels)

            outputs = torch.argmax(outputs, axis=1)
            valid_running_loss += loss.item()

            correct += (outputs == labels).sum().item()

        valid_running_loss /= len(valid_loader.dataset)
        valid_acc = correct / len(valid_loader.dataset)

        print(f"epoch: {epoch + 1} valid Loss: {valid_running_loss}")
        print(f"epoch: {epoch + 1} valid Acc: {valid_acc}")
        valid_epoch_loss += [valid_running_loss]
        valid_epoch_acc += [valid_acc]

    # test
    model.eval()
    test_running_loss = 0.0
    correct = 0
    d_correct = 0
    c_correct = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            imgs, labels = [t.to(device) for t in data]

            outputs = model(imgs)
            loss = loss_func(outputs, labels)

            outputs = torch.argmax(outputs, axis = 1)
            test_running_loss += loss.item()

            correct += (outputs == labels).sum().item()
            d_correct += (outputs == labels).logical_and(labels).sum().item()
            c_correct += (outputs == labels).logical_and(torch.logical_not(labels)).sum().item()

        test_running_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)

        print(f"epoch: {epoch + 1} test Loss: {test_running_loss}")
        print(f"epoch: {epoch + 1} test Acc: {acc},  dog acc: {d_correct/1245}, cat acc: {c_correct/1255}")
        test_epoch_loss += [test_running_loss]
        epoch_acc += [acc]


    writer.add_scalars('Loss', {'train':train_running_loss,
                                'valid':valid_running_loss,
                                'test':test_running_loss}, epoch + 1)
    writer.add_scalars('Acc', {'valid':valid_acc,
                                'test':acc}, epoch + 1)
writer.close()
