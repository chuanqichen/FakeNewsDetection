import sys
import os
resnet_dir = os.path.join(os.path.dirname(__file__), '../resnet/')
sys.path.append(resnet_dir)
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from transformers import BertForSequenceClassification
from my_resnet import resnet50_2way
from FakedditDataset import FakedditHybridDataset, my_collate
from HybridModel import LateFusionModel
from training import ModelTrainer

# Load bert model
bert_classifier = BertForSequenceClassification.from_pretrained('../bert_save_dir')

# Load resnet
resnet_model = resnet50_2way(pretrained=False)
# resnet_dict = torch.load('../resnet/resnet_epc20.pt', map_location=torch.device('cpu'))
resnet_dict = torch.load('../resnet/resnet_epc20.pt')
resnet_model.load_state_dict(resnet_dict)

# Create fusion model
hybrid_model = LateFusionModel(resnet_model, bert_classifier)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
hybrid_model = hybrid_model.to(device)

# Prepare datesets
csv_dir = "../../Data/"
img_dir = "../../Data/public_image_set/"
l_datatypes = ['train', 'validate', 'test']
csv_fnames = {
    'train': 'multimodal_train.tsv',
    'validate': 'multimodal_validate.tsv',
    'test': 'multimodal_test_public.tsv'
}
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
hybrid_datasets = {x: FakedditHybridDataset(os.path.join(csv_dir, csv_fnames[x]), img_dir, transform=data_transforms)
                   for x in l_datatypes}
dataset_sizes = {x: len(hybrid_datasets[x]) for x in l_datatypes}

# Dataloader
dataloaders = {x: torch.utils.data.DataLoader(hybrid_datasets[x], batch_size=64, shuffle=True, num_workers=2,
                                              collate_fn=my_collate) for x in l_datatypes}

# Specify loss function
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

# Specify optimizer
optimizer_ft = optim.Adam(hybrid_model.parameters(), lr=1e-4)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Trainer isntance
trainer = ModelTrainer(l_datatypes, hybrid_datasets, dataloaders, hybrid_model)
# Train the model
trainer.train_model(criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1, report_len=1000)
trainer.save_model('hybrid_model.pt')

#######################
#       Testing
#######################
trainer.generate_eval_report('hybrid_report.json')
