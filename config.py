from imports import *

class config:
    def __init__(self, vit=0, num_classes=2, gamma = 0.1, step_size = 3, lr = 0.0001,
                 image_size = 160, num_epochs=7, batch_size=32, train_resnet=False, train_deit = True):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.gamma = gamma
        self.step_size = step_size
        self.num_classes = num_classes

        self.train_resnet = train_resnet
        self.train_deit = train_deit
        self.vit = vit