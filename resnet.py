from imports import *

class resnet:
    def __init__(self, pretrained=True, num_classes=2):
        self.pretrained = pretrained
        self.num_classes = num_classes

    def build_resnet(self):
        resnet = models.resnet152(pretrained=self.pretrained) # resnet152
        num_ftrs = resnet.fc.in_features
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        resnet.fc = nn.Linear(num_ftrs, self.num_classes)
        return resnet