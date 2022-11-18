"""°°°
# My very first Image Classifier
> [aemonge](https://www.aemonge.com)
°°°"""
# |%%--%%| <JEszbIJb4l|jYyKGfUsv5>

# %matplotlib inline # pyright ignore
# %config InlineBackend.figure_format = 'retina'

from typing import OrderedDict
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import helper

EPOCHS = 10
LEARNING_RATE = 0.003
USE_PRELEARNED = True
USE_GPU = False

# |%%--%%| <jYyKGfUsv5|3iz545WjRB>

###
data_dir = './pets_images/train'
test_data_dir = './pets_images/test'

transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
trains_data_set = datasets.ImageFolder(data_dir, transform = transform)
dataloader = DataLoader(trains_data_set, batch_size=32, shuffle=True)
# Run this to test your data loader
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)

##
data_dir = './pets_images'
mean_norm = [0.485, 0.456, 0.406]
std_norm  = [0.229, 0.224, 0.225]
expected_size = (224, 224)

train_transforms = transforms.Compose([
    transforms.RandomRotation(44),
    transforms.RandomResizedCrop(203),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(134),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_norm, std_norm)
])
test_transforms = transforms.Compose([
    transforms.Resize(expected_size),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(data_dir + "/train", transform = train_transforms)
test_data = datasets.ImageFolder(data_dir + "/test", transform = test_transforms)

trainloader = DataLoader(train_data, batch_size=32)
testloader  = DataLoader(test_data,  batch_size=32)

# |%%--%%| <3iz545WjRB|jorKvq5nP1>

data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)

# |%%--%%| <jorKvq5nP1|CI6M6wLGoE>

data_iter = iter(trainloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)

# |%%--%%| <CI6M6wLGoE|KtEyaMs6dW>

mymodel = nn.Sequential(
    # nn.Linear(784, 128), nn.ReLU(), dropout(),
    nn.Linear(784, 128), nn.Softshrink(), nn.Dropout(0.18),
    nn.Linear(128,  64), nn.ReLU(), nn.Dropout(0.18),
    nn.Linear( 64,  64), nn.ReLU(), nn.Dropout(0.18),
    nn.Linear( 64,  32), nn.ReLU(), nn.Dropout(0.18),
    nn.Linear( 32, 128), nn.Softmax(), nn.Dropout(0.18),
    nn.Linear(128,  10), nn.LogSoftmax(dim=1), # output on the densenet is 1_000
)
if USE_PRELEARNED:
    model = models.densenet121(pretrained=True) # OR VGGNet
else:
    model = mymodel

# Statistics
# ps = torch.exp(model(images))
# top_p, top_class = ps.topk(1, dim=1)
# equals = top_class == labels.view(*top_class.shape)
# accuracy = torch.mean(equals.type(torch.FloatTensor))

# Freeze our Feat Params, aka no updates nor backtracking.
if USE_PRELEARNED:
    for p in model.parameters():
        p.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

if USE_GPU:
    model.cuda()
else:
    model.cpu()


criterion = nn.NLLLoss()
optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

train_losses, test_losses = [], []
for e in range(EPOCHS):
    running_loss = 0
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        model.eval()
        # print(f'Accuracy: {accuracy.item()*100}%')

        t_running_loss = 0
        test_correct = 0
        with torch.no_grad():
            for images, labels in testloader:
                t_log_ps = model(images)
                t_loss = criterion(t_log_ps, labels)

                t_running_loss += t_loss.item()

                ps = torch.exp(t_log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_correct += equals.sum().item()

        print(f"Epoch: { e + 1 }/{EPOCHS}\n",
              "Training Loss: {:.3f}.. ".format(running_loss),
              "Test Loss: {:.3f}.. ".format(t_running_loss),
              "Test Accuracy: {:.3f}".format(test_correct / len(testloader.dataset)))

# |%%--%%| <KtEyaMs6dW|hc3wyO17Ek>
"""°°°
 🍾 ⭐ Kudos !!!
°°°"""