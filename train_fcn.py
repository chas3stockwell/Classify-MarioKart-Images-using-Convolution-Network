import torch
import numpy as np
import gc

from .models import FCN, ClassificationLoss, save_model, DenseClassificationLoss, ResidualBlock, FocalLoss
from .utils import load_dense_data,  DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
from torchvision import transforms
import torch.utils.tensorboard as tb
import torch.nn as nn
import sklearn.utils.class_weight as class_weight

def train(args):
    from os import path
    
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    num_classes = 5
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.01

    #model = FCN(ResidualBlock, [1, 1, 1, 1]).to(device)
    model = FCN(3, 5).to(device)
    # Loss and optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)  
    optimizer = torch.optim.SGD(model.parameters(), lr=0.016, momentum=0.92, weight_decay=1e-4)
    
    con = ConfusionMatrix()

    

    dense_transform_norm = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.32343397, 0.33100516, 0.34438375], std=[0.16127683, 0.13571456, 0.16258068]),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.ColorJitter(brightness=0.75, contrast=0.75, saturation=1, hue=0.25)
    ])

    
    print(torch.device)
    dataset = load_dense_data('dense_data/dense_data/train',  num_workers=0, batch_size=batch_size)
    v_dataset = load_dense_data('dense_data/dense_data/valid',   num_workers=0, batch_size=batch_size)
    print(len(dataset))
    #dl = DenseClassificationLoss()
    tens = torch.ones(5)
    class_dist = torch.tensor([0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]) #these are weights for dense

    #print(class_weights) #([1.0000, 1.0000, 4.0000, 1.0000, 0.5714])

    #dl = torch.nn.CrossEntropyLoss(weight= class_weights)
    #dl = FocalLoss(1.5, alpha=torch.tensor([0.2, 1.25, 0.2, 3, 2.75]))
    #print(-1*torch.log(class_weights))

    weight = torch.tensor([0.2, 1.25, 0.2, 3, 2.75])
    mean, std, var = torch.mean(weight), torch.std(weight), torch.var(weight)
    weight_normalize = (weight - mean) / std
    intercept_weight = torch.tensor([0.01, 0.05, 0.02, 0.46, 0.46])
    #dl = torch.nn.CrossEntropyLoss(weight=intercept_weight)
    
    dl = ClassificationLoss()

    
    #dl = torch.nn.CrossEntropyLoss(weight=-(torch.ones(5) - class_weights)**2 * torch.log(class_weights))
    #dl = torch.nn.CrossEntropyLoss(weight=alpha_adjusted)
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataset): 
            #print(" Epoch : ", epoch, "iteration : ", i)
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.type(torch.LongTensor) #convert to LongTensor
            labels = labels.to(device)
            
            
            outputs = model(images)
            ''' 
            if i == 0:
                con._make(outputs.argmax(1), labels )
            else:
                con.add( outputs.argmax(1), labels )
            '''
            loss = dl(outputs, labels)
            
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
            
            for i, (images, labels) in enumerate(v_dataset):
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
        

            #print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 
        print(con.iou)
        print(con.class_iou)
        



    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
