from torchvision import models
import torch.onnx
import torch.nn as nn

class NN_Model(nn.Module):
    def __init__(self, model_name, num_classes=7, use_pretrained=True):
        super(NN_Model, self).__init__()
        
        self.use_pretrained = use_pretrained
        self.input_size = 224

        if model_name == 'resnet50':
            self.model = ResNet(num_classes) 
        elif model_name == 'vgg':
            self.model = Vgg(num_classes)
        elif model_name == 'inception':
            self.model = InceptionV3(num_classes)
            self.input_size = 299
        else:
            raise ValueError(f"Invalid model_name: {model_name}")
    
    def forward(self, x):
        return self.model(x)
    
class ResNet(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(ResNet, self).__init__()
        # Define your model architecture for model1 here
        self.features = models.resnet18(pretrained=use_pretrained)
        num_features = self.features.fc.in_features
        self.features.fc = nn.Linear(num_features, num_classes)
        self.input_size = 224
        
    def forward(self, x):
        return self.features(x)

class Vgg(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(Vgg, self).__init__()
        self.features = models.vgg16(pretrained=use_pretrained)
        num_features = self.features.classifier[6].in_features
        self.features.classifier[6] = nn.Linear(num_features, num_classes)


    def forward(self, x):
        return self.features(x)

class InceptionV3(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(InceptionV3, self).__init__()
        # Define your model architecture for model3 here
        self.features = models.inception_v3(pretrained=use_pretrained)
        # get num_features from the last layer of the model
        num_features = self.features.AuxLogits.fc.in_features
        self.features.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.features(x)

