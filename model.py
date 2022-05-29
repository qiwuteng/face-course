import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.init as torch_init
import config as cfg

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def Classifier(model, num_classes=2):
    if model in  cfg.model_input_dim.keys():
        input_dim = cfg.model_input_dim[model]
    
    if 'AlexNet' in model or 'VGG' in model:
        classifier = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    elif 'ResNet' in model:
        classifier = nn.Linear(input_dim, num_classes)

    classifier.apply(weights_init)
    return classifier

class Model(nn.Module):
    def __init__(self, model, pretrained):
        super(Model, self).__init__()

        if model == "AlexNet":
            self.model = models.alexnet(pretrained=pretrained)
        elif model == "VGG16":
            self.model = models.vgg16(pretrained=pretrained)
        elif model == "VGG19":
            self.model = models.vgg16(pretrained=pretrained)
        elif model == "ResNet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model == "ResNet50":
            self.model = models.resnet50(pretrained=pretrained)

        self.features = nn.Sequential(*list(self.model.children())[:-1])

        self.hair = Classifier(model)
        self.gender = Classifier(model)
        self.earring = Classifier(model)
        self.smile = Classifier(model)
        self.frontal_face = Classifier(model)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        hair = self.hair(x)
        gender = self.gender(x)
        earring = self.earring(x)
        smile = self.smile(x)
        frontal_face = self.frontal_face(x)
        return hair, gender, earring, smile, frontal_face

if __name__ == "__main__":
    a = Model('AlexNet', True)
    print(a.hair[6].weight)
    print(a.gender[6].weight)
    # print(a)