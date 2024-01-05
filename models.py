import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) 

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        c = torch.nn.CrossEntropyLoss()
        loss = c(input, target)
        return loss

class DenseClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        class_weights = torch.tensor([0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153])
        c = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean').to(device)
        loss = c(input, target)
        return loss
        
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class Block(nn.Module): #simple up-channel block 
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class CNNClassifier(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[3, 4, 6, 3], num_classes = 6):
        super(CNNClassifier, self).__init__()
        self.inplanes = 12
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 12, kernel_size = 7, stride = 2, padding = 3), #going to change all 64s to 128
                        nn.BatchNorm2d(12),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 24, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 48, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 96, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 192, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.lin1 = nn.Linear(768, 120)
        self.lin2 = torch.nn.Linear(120, 84)
        self.lin3 = torch.nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.normalize = transforms.Normalize(mean=[0.32343397, 0.33100516, 0.34438375], std=[0.16127683, 0.13571456, 0.16258068])

        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.normalize(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)
        
        return x
            

class FCNb(nn.Module):       #original layer values - 3, 4, 6, 3
    def __init__(self, block=ResidualBlock, layers=[3, 4, 6, 3], num_classes = 5):
        super(FCNb, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 1, padding = 3), #going to change all 64s to 128
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        #self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        #self.layer1 = self._make_layer(block, 128, layers[1], stride = 1)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 1)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 1)
        self.layer3b = self._make_layer(block, 256, layers[2], stride = 1)
        #self.layer2b = self._make_layer(block, 128, layers[1], stride = 1)
        self.layer1b = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer0b = self._make_layer(block, num_classes, layers[0], stride = 1)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):

        
        #print("Original shape: ", x.shape)
        x = self.conv1(x)
        x = self.maxpool(x)
        #print("First conv1 layer: ", x.shape)

        #x = self.layer1(x)
        #print("Layer1", x.shape)
        
        x = self.layer2(x)
        #print("Layer2", x.shape)

        #x = self.layer3(x)
        #print("Bottom of U layer ", x.shape)

        x = self.layer3b(x)
        #print("Layer3b", x.shape)

        #x = self.layer2b(x)
        #print("Layer2b", x.shape)

        x = self.layer1b(x)
        #print("Layer1b", x.shape)

        x = self.layer0b(x)
        #print("Layer0b", x.shape)
        #print("Final Shape: ", x.shape)

        #print("")

        #this needs to end [100, 5, 3, 4]
        return x
        
       
        
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """




def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        
        if x.shape[2] == 1 or x.shape[3] == 1:
            return self.conv(x)
        else: 
            return self.maxpool_conv(x)
        
       


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) #was stride=2
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) #was kernel_size=1

    def forward(self, x):
        return self.conv(x)
    

class FCN(nn.Module):
    def __init__(self, n_channels=3, n_classes=5, bilinear=False):
        super(FCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16))
        #self.down1 = (Down(8, 16))
        self.down2 = (Down(16, 32))
        self.down3 = (Down(32, 64))
        factor = 2 if bilinear else 1
        self.down4 = (Down(64, 128 // factor))
        self.up1 = (Up(128, 64 // factor, bilinear))
        self.up2 = (Up(64, 32 // factor, bilinear))
        self.up3 = (Up(32, 16 // factor, bilinear))
        #self.up4 = (Up(16, 8 // factor, bilinear))
        self.outc = (OutConv(16, n_classes))

        self.normalize = transforms.Normalize(mean=[0.32343397, 0.33100516, 0.34438375], std=[0.16127683, 0.13571456, 0.16258068])

    def forward(self, x): #time for some padding debugging
        x = self.normalize(x)
        x1 = self.inc(x)
        #print(" Original Convolution: ", x1.shape)
        #x2 = self.down1(x1)
        #print("  Down one: ", x2.shape)
        x3 = self.down2(x1)
        #print("   Down 2: ", x3.shape)
        x4 = self.down3(x3)
        #print("    Down 3: ", x4.shape)
        x5 = self.down4(x4)
        #print("    Down 3: ", x4.shape)
        x = self.up1(x5, x4)
        #print("     Up 1: ", x4.shape)
        x = self.up2(x, x3)
        #print("      Up 2: ", x4.shape)
        x = self.up3(x, x1)
        #print("       Up 3: ", x4.shape)
        #x = self.up4(x, x1)
        #print("        Up 4: ", x4.shape)
        logits = self.outc(x)
        #print("Final Shape: ", logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
    
}