import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from FakedditDataset import FakedditDataset, my_collate
import os, time, copy
from tqdm import tqdm
from collections import deque
from statistics import mean
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

csv_dir = "../../Data/"
img_dir = "../../Data/public_image_set/"
l_datatypes = ['train', 'validate']
csv_fnames = {'train': 'multimodal_train.tsv', 'validate': 'multimodal_validate.tsv'}
image_datasets = {x: FakedditDataset(os.path.join(csv_dir, csv_fnames[x]), img_dir, transform=data_transforms) for x in
                  l_datatypes}
# Dataloader, pin_memory doesn't make a difference
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=2, collate_fn=my_collate) for x in l_datatypes}

dataset_sizes = {x: len(image_datasets[x]) for x in l_datatypes}
# class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Note: corrupted images will be skipped in training")

def train_model(model, criterion, optimizer, scheduler, num_epochs=2, report_len=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in l_datatypes:
            print(f'{phase} phase')
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Create a queue to monitor loss
            loss_q = deque(maxlen=report_len)
            acc_q = deque(maxlen=report_len)
            # Iterate over data.
            counter = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                counter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(f"output shape: {outputs.size()}; target shape: {labels.size()}")
                    # _, preds = torch.max(outputs, 1)
                    #t_pred = outputs > 0.5
                    output = torch.nn.functional.softmax(outputs[0], dim=0)
                    t_pred = output > 0.5
                    acc = (t_pred.squeeze() == labels).float().sum() / len(labels)
                    acc_q.append(acc.item())
                    loss = criterion(outputs, labels.unsqueeze(-1).float())
                    loss_q.append(loss.item())
                    if counter % report_len == 0:
                        print(f"Iter {counter}, loss: {mean(loss_q)}, accuracy:{mean(acc_q)}")
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(t_pred.squeeze() == labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Initialize model and optimizer
#model_ft = models.resnet18(pretrained=True)
#model_ft = models.resnet50(pretrained=True)
model_ft = models.inception_v3(pretrained=True)
set_parameter_requires_grad(model_ft, True)   # freeze the pretrained model
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 1.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)

# save model
torch.save(model_ft.state_dict(), 'fakeddit_inception_epochs2.pt')

torch.save(model_ft, "inception_model_save_epochs2")
