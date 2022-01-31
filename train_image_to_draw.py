from imports import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import resnet

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

def load_drawings_network(exp, draw_checkpoint_dir, device):
    model_resnet = resnet(pretrained=True, num_classes=10).build_resnet().to(device)
    filename = os.path.join(draw_checkpoint_dir, exp + '_best_model' + '.pth.tar')
    checkpoint = torch.load(filename, map_location="cpu")
    model_resnet.load_state_dict(checkpoint)
    return model_resnet

def train_image_network_as_draw(num_epochs, arch_type, image_model, device, dataloaders_train, dataloaders_test,
                                optimizer, criterion, scheduler, dataset_sizes, exp, checkpoint_dir, draw_checkpoint_dir):
# def train_image_network_as_draw(num_epochs, arch_type, image_model, device, dataloaders,
#                                 optimizer, criterion, scheduler, dataset_sizes, exp, checkpoint_dir, draw_checkpoint_dir):
    drawing_model = load_drawings_network(exp, draw_checkpoint_dir, device)
    train_losses = []
    train_accuracy = []
    dist_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    for epoch in range(num_epochs):
        print('Epoch {}/{} for {}'.format(epoch, num_epochs - 1, arch_type))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                image_model.train()  # Set model to training mode
                dataloaders = dataloaders_train
            else:
                image_model.eval()  # Set model to evaluate mode
                dataloaders = dataloaders_test
            drawing_model.eval()
            running_loss = 0.0
            running_dist_loss = 0.0
            running_cross_entropy_loss = 0.0
            running_corrects = 0
            i = 0
            # Iterate over data.
            for draw_data, image_data in dataloaders:
                draw_inputs, draw_labels = draw_data
                image_inputs, image_labels = image_data

                draw_inputs = draw_inputs.to(device)
                # draw_labels = draw_labels.to(device)
                image_inputs = image_inputs.to(device)
                image_labels = image_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                i += 1
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = image_model(image_inputs)
                    if "vit" in arch_type:
                        _, preds = torch.max(outputs, 1)
                    else:
                        _, preds = torch.max(outputs, 1)
                    cross_entropy_loss = criterion(outputs, image_labels)

                    embedding_drawing_model = torch.nn.Sequential(*(list(drawing_model.children())[:-1]))
                    draw_outputs = embedding_drawing_model[:-1](draw_inputs)
                    embedding_image_model = torch.nn.Sequential(*(list(image_model.children())[:-1]))
                    image_outputs = embedding_image_model[:-1](image_inputs)
                    dist_loss = -dist_criterion(draw_outputs, image_outputs).mean()
                    dist_loss = torch.add(dist_loss, 1)
                    # if epoch >= 10:
                    #     loss = dist_loss * 0.1 + cross_entropy_loss * 0.9
                    # else:
                    #     loss = dist_loss * 0.99 + cross_entropy_loss * 0.01
                    loss = dist_loss * 0.8 + cross_entropy_loss * 0.2
                    # loss = dist_loss
                    # loss = cross_entropy_loss
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * image_inputs.size(0)
                running_dist_loss += dist_loss.item() * image_inputs.size(0)
                running_cross_entropy_loss += cross_entropy_loss.item() * image_inputs.size(0)
                running_corrects += torch.sum(preds == image_labels.data)

            if phase == 'train':
                scheduler.step()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_dist_loss = running_dist_loss / dataset_sizes[phase]
            epoch_cross_entropy_loss = running_cross_entropy_loss / dataset_sizes[phase]
            epoch_acc = 100 * running_corrects.float() / dataset_sizes[phase]
            # acc = 100 * (animal_accuracy + not_animal_accuracy) / (total_animal + total_not_animal)
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracy.append(epoch_acc.item())
                print('{} Loss: {:.4f}, dist_loss: {:.4f}, cross_entropy_loss: {:.4f}, acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_dist_loss, epoch_cross_entropy_loss, epoch_acc))
            if phase == 'test':
                total_loss = running_loss / dataset_sizes['test']
                total_dist_loss = running_dist_loss / dataset_sizes['test']
                total_cross_entropy_loss = running_cross_entropy_loss / dataset_sizes['test']
                acc = 100 * running_corrects.float() / dataset_sizes['test']
                if epoch == 0:
                    best_acc = acc
                if acc >= best_acc:
                    best_acc = acc
                    print("new best acc {}".format(best_acc))
                    filename = os.path.join(checkpoint_dir, exp + '_best_model' + '.pth.tar')
                    torch.save(image_model.state_dict(), filename)
                    best_model_wts = copy.deepcopy(image_model.state_dict())
                print("acc is {} loss is {} dist loss is {} cross entropy loss is {}".format(acc, total_loss, total_dist_loss, total_cross_entropy_loss))

    filename = os.path.join(checkpoint_dir, exp + '_best_model' + '.pth.tar')
    checkpoint = torch.load(filename, map_location="cpu")
    # self.model.load_state_dict(checkpoint["state_dict"], strict=False)
    image_model.load_state_dict(checkpoint)
    return image_model