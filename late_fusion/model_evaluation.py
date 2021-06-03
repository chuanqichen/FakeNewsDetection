import sys
import os
resnet_dir = os.path.join(os.path.dirname(__file__), '../resnet/')
sys.path.append(resnet_dir)
import torch
from torchvision import transforms
from transformers import BertForSequenceClassification, BertConfig
from my_resnet import resnet50_2way
from FakedditDataset import FakedditHybridDataset, my_collate
from HybridModel import LateFusionModel
from training import ModelTrainer

# Load bert model
bert_classifier = BertForSequenceClassification.from_pretrained('../bert_save_dir')

# Load resnet
resnet_model = resnet50_2way(pretrained=False)
# resnet_dict = torch.load('../resnet/fakeddit_resnet.pt')
# resnet_model.load_state_dict(resnet_dict)

# Create fusion model
hybrid_model = LateFusionModel(resnet_model, bert_classifier)
hybrid_dict = torch.load('hybrid_model_run1.pt')
hybrid_model.load_state_dict(hybrid_dict)
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

# Dataloader
dataloaders = {x: torch.utils.data.DataLoader(hybrid_datasets[x], batch_size=64, shuffle=True, num_workers=2,
                                              collate_fn=my_collate) for x in l_datatypes}

# Create trainer isntance
trainer = ModelTrainer(l_datatypes, hybrid_datasets, dataloaders, hybrid_model)

# Evaluate on test set
print("Evaluating model on test set...")
trainer.generate_eval_report('report.json')