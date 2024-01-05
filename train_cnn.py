from .models import CNNClassifier, save_model, ClassificationLoss, ResidualBlock
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import numpy as np
import torch.nn as nn
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    from os import path

    num_classes = 6
    num_epochs = 20
    batch_size = 100
    learning_rate = 0.01

    model = CNNClassifier(ResidualBlock, [3, 4, 6, 3]).to(device)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.92, weight_decay=1e-4)  

    # Train the model
    '''total_step = len(train_loader)
    total_step = len(train_loader)'''
    import gc

    
    print(torch.cuda.is_available())

    con = ConfusionMatrix()
    color = T.ColorJitter(0.8, 0.3), T.RandomCrop(32), T.CenterCrop(size=32)

    train_trans = T.Compose( [T.ColorJitter(0.8, 0.3), T.RandomHorizontalFlip(0.5), T.ToTensor()]) # 96
    val_trans = T.Compose( [T.ToTensor()])

    dataset = load_data('data/train', num_workers=0, batch_size=batch_size, transform=train_trans)
    v_dataset = load_data('data/valid', num_workers=0, batch_size=batch_size, transform=val_trans)



    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataset):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, loss.item()))
                
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in v_dataset:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                if i == 0:
                    con._make(outputs.argmax(1), labels )
                else:
                    con.add(  outputs.argmax(1), labels )
           
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            print(correct)
            print(total)
        
        
            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 
        #print(con.iou)
        #print(con.class_iou)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
