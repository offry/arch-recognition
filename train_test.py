from imports import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def print_graph(title, xlabel, ylabel, plot, accuracy=0, plot2=0, plot3=0, plot4=0, plot5=0, plot6=0):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    if(accuracy==1):
        plt.plot(plot, label="Train accuracy")
        plt.plot(plot2, label="Train accuracy for animal images")
        plt.plot(plot3, label="Train accuracy for not animal images")
        plt.plot(plot4, label="Test accuracy")
        plt.plot(plot5, label="Test accuracy for animal images")
        plt.plot(plot6, label="Test accuracy for not animal images")
    else:
        plt.plot(plot, label="Train loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()



class train_test:

    def __init__(self, model, criterion, optimizer, scheduler, num_epochs, dataloaders, device, dataset_sizes, exp, class_names, checkpoint_dir, model_name):
        self.criterion = criterion
        # self.criterion = torch.nn.TripletMarginLoss(margin=1)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.device = device
        self.dataset_sizes = dataset_sizes
        self.model_name = model_name
        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.exp = exp
        self.class_names = class_names
        self.checkpoint_dir = checkpoint_dir

    def train_model(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        train_losses = []
        train_accuracy = []
        # self.criterion = FocalLoss(gamma=1)
        for epoch in range(self.num_epochs):
            print('Epoch {}/{} for {}'.format(epoch, self.num_epochs - 1, self.model_name))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                i = 0
                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    # negatives, positives = [], []
                    # for j in range(len(inputs)):
                    #     for k in range(len(inputs)):
                    #         if labels[j]==labels[k] and j!=k:
                    #             positives.append(inputs[k])
                    #             break
                    #         if k==len(inputs)-1:
                    #             positives.append(inputs[j])
                    # for j in range(len(inputs)):
                    #     for k in range(len(inputs)):
                    #         if labels[j] != labels[k] and j != k:
                    #             negatives.append(inputs[k])
                    #             break
                    # positives = torch.stack(positives)
                    # negatives = torch.stack(negatives)
                    # positives = positives.to(self.device)
                    # negatives = negatives.to(self.device)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # if i % 50 == 0:
                    #     print('epoch {} num of batches trained on: {} out of {}'.format(epoch, i,
                    #                                                                     len(dataloaders[phase])))
                    # forward
                    # track history if only in train
                    i += 1
                    with torch.set_grad_enabled(phase == 'train'):
                        # for input, positive, negative in (zip(inputs, negatives, positives)):
                        outputs = self.model(inputs)
                        # outputs_positive = self.model(positives)
                        # outputs_negative = self.model(negatives)
                        if "vit" in self.model_name:
                            _, preds = torch.max(outputs, 1)
                        else:
                            _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        # loss = self.criterion(outputs, outputs_positive, outputs_negative)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = 100 * running_corrects.float() / self.dataset_sizes[phase]
                # acc = 100 * (animal_accuracy + not_animal_accuracy) / (total_animal + total_not_animal)
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracy.append(epoch_acc.item())
                    print('{} Loss: {:.4f}, {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
                if phase=='test':
                    total_loss = running_loss / self.dataset_sizes['test']
                    acc = 100 * running_corrects.float() / self.dataset_sizes['test']
                    if epoch==0:
                        best_acc = acc
                    if acc>=best_acc:
                        best_acc = acc
                        print("new best acc {}".format(best_acc))
                        filename = os.path.join(self.checkpoint_dir, self.exp + '_best_model' + '.pth.tar')
                        torch.save(self.model.state_dict(), filename)
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                    print("acc is {} and loss is {}".format(acc, total_loss))
                # if epoch%3==0:
                #     filename = self.exp + '_checkpoint_' + str(epoch) + '.pth.tar'
                #     torch.save(self.model.state_dict(), filename)
        filename = os.path.join(self.checkpoint_dir, self.exp + '_best_model' + '.pth.tar')
        checkpoint = torch.load(filename, map_location="cpu")
        # self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.load_state_dict(checkpoint)
        return self.model