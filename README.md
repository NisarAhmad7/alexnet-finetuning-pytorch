# AlexNet Image Classifier (Finetuning)

This project demonstrates how to finetune a pretrained AlexNet model on a custom image dataset using PyTorch.

##  Features
- Uses pretrained AlexNet (ImageNet)
- Finetuned on custom dataset (e.g., cat vs dog)
- Data augmentation for better performance
- Simple and clean PyTorch training pipeline

##  Project Structure


.
├── train.py
├── requirements.txt
├── README.md
└── data/
└── train/
├── class1/
└── class2/


##  Installation

```bash
pip install -r requirements.txt
-> Dataset Structure

Your dataset should be organized like this:

data/train/
    cat/
        img1.jpg
        img2.jpg
    dog/
        img3.jpg
        img4.jpg
-> Run Training
python train.py
-> Model Details
Model: AlexNet (pretrained on ImageNet)
Task: Image Classification
Loss: CrossEntropyLoss
Optimizer: Adam
-> Data Augmentation
Resize
Random Crop
Horizontal Flip
Normalization (ImageNet)
-> Output

The finetuned model will be saved as:

alexnet_finetuned.pth
-> Tech Stack
Python
PyTorch
Torchvision