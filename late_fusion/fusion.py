import torch
from torchvision import transforms
from transformers import BertForSequenceClassification
from resnet.my_resnet import resnet50_2way
from resnet.FakedditDataset import FakedditHybridDataset, my_collate
import os

# Load bert model
bert_model = BertForSequenceClassification.from_pretrained('../bert_save_dir')
print(bert_model)

# Load resnet
resnet_model = resnet50_2way(pretrained=False)
resnet_dict = torch.load('../fakeddit_resnet.pt', map_location=torch.device('cpu'))
resnet_model.load_state_dict(resnet_dict)
print(resnet_model)

# Create fusion model


# Prepare datesets
csv_dir = "../../Data/"
img_dir = "../../Data/public_image_set/"
l_datatypes = ['train', 'validate']
csv_fnames = {'train': 'multimodal_train.tsv', 'validate': 'multimodal_validate.tsv'}
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
hybrid_datasets = {x: FakedditHybridDataset(os.path.join(csv_dir, csv_fnames[x]), img_dir, transform=data_transforms) for x in
                   l_datatypes}
dataset_sizes = {x: len(hybrid_datasets[x]) for x in l_datatypes}

# Dataloader
dataloaders = {x: torch.utils.data.DataLoader(hybrid_datasets[x], batch_size=64, shuffle=True, num_workers=2,
                                              collate_fn=my_collate) for x in l_datatypes}
