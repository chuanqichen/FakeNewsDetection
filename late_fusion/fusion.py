import sys
import os
resnet_dir = os.path.join(os.path.dirname(__file__), '../resnet/')
sys.path.append(resnet_dir)
import torch
from torchvision import transforms
from transformers import BertForSequenceClassification
from my_resnet import resnet50_2way
from FakedditDataset import FakedditHybridDataset, my_collate
from HybridModel import LateFusionModel

# Load bert model
bert_classifier = BertForSequenceClassification.from_pretrained('../bert_save_dir')
#print(bert_classifier)

# Load resnet
resnet_model = resnet50_2way(pretrained=False)
resnet_dict = torch.load('../resnet/fakeddit_resnet.pt', map_location=torch.device('cpu'))
resnet_model.load_state_dict(resnet_dict)
#print(resnet_model)

# Create fusion model
hybrid_model = LateFusionModel(resnet_model, bert_classifier)
#print(hybrid_model._bert.config.return_dict)
#print(hybrid_model._bert)
#print(hybrid_model._resnet)

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

# Test
test_in = next(iter(dataloaders['train']))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
for key in test_in:
    test_in[key].to(device)
hybrid_model(test_in)
