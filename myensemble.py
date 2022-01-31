from imports import *

class myensemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=2):
        super(myensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        resnet = models.resnet50(pretrained=True)
        resnet_num_ftrs = resnet.fc.in_features
        # Create new classifier
        # self.fc = nn.Linear(resnet_num_ftrs + 768, nb_classes)
        # self.fc2 = nn.Linear(resnet_num_ftrs, nb_classes)
        self.fc = nn.Linear(nb_classes, nb_classes)

    def forward(self, x):
        out1 = self.modelA(x).last_hidden_state
        out2 = self.modelB(x)
        out = out1 + out2
        x = self.fc1(out)
        return torch.softmax(x, dim=1)