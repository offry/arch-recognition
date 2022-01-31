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

def train_image_network_as_draw(num_epochs, arch_type, image_model, drawing_model, device, dataloaders_train, dataloaders_test,
                                optimizer_image, optimizer_draw, criterion, scheduler_image, scheduler_draw, dataset_sizes, exp,
                                image_checkpoint_dir, draw_checkpoint_dir):

    train_losses_image, train_losses_draw = [], []
    train_accuracy_image, train_accuracy_draw = [], []
    dist_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    for epoch in range(num_epochs):
        print('Epoch {}/{} for {}'.format(epoch, num_epochs - 1, arch_type))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                image_model.train()  # Set model to training mode
                dataloaders = dataloaders_train
                drawing_model.train()
            else:
                image_model.eval()  # Set model to evaluate mode
                dataloaders = dataloaders_test
                drawing_model.eval()
            running_loss_image, running_loss_draw = 0.0, 0.0
            running_dist_loss = 0.0
            running_cross_entropy_loss_image, running_cross_entropy_loss_draw = 0.0, 0.0
            running_corrects_image, running_corrects_draw = 0, 0
            i = 0
            # Iterate over data.
            for draw_data, image_data in dataloaders:
                draw_inputs, draw_labels = draw_data
                image_inputs, image_labels = image_data

                draw_inputs = draw_inputs.to(device)
                draw_labels = draw_labels.to(device)
                image_inputs = image_inputs.to(device)
                image_labels = image_labels.to(device)

                # zero the parameter gradients
                optimizer_image.zero_grad()
                optimizer_draw.zero_grad()

                i += 1
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_image_classification = image_model(image_inputs)
                    outputs_draw_classification = drawing_model(draw_inputs)

                    if "vit" in arch_type:
                        _, preds_image = torch.max(outputs_image_classification, 1)
                        _, preds_draw = torch.max(outputs_draw_classification, 1)

                    else:
                        _, preds_image = torch.max(outputs_image_classification, 1)
                        _, preds_draw = torch.max(outputs_draw_classification, 1)

                    cross_entropy_loss_image = criterion(outputs_image_classification, image_labels)
                    cross_entropy_loss_draw = criterion(outputs_draw_classification, draw_labels)

                    embedding_drawing_model = torch.nn.Sequential(*(list(drawing_model.children())[:-1]))
                    draw_outputs = embedding_drawing_model[:-1](draw_inputs)
                    embedding_image_model = torch.nn.Sequential(*(list(image_model.children())[:-1]))
                    image_outputs = embedding_image_model[:-1](image_inputs)
                    dist_loss = -dist_criterion(draw_outputs, image_outputs).mean()
                    dist_loss = torch.add(dist_loss, 1)

                    loss_image = dist_loss * 0.75 + cross_entropy_loss_image * 0.25
                    loss_draw = dist_loss * 0.75 + cross_entropy_loss_draw * 0.25

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_image.backward()
                        optimizer_image.step()
                        loss_draw.backward()
                        optimizer_draw.step()

                # statistics
                running_loss_image += loss_image.item() * image_inputs.size(0)
                running_loss_draw += loss_draw.item() * draw_inputs.size(0)

                running_dist_loss += dist_loss.item() * image_inputs.size(0)

                running_cross_entropy_loss_image += cross_entropy_loss_image.item() * image_inputs.size(0)
                running_cross_entropy_loss_draw += cross_entropy_loss_draw.item() * draw_inputs.size(0)

                running_corrects_image += torch.sum(preds_image == image_labels.data)
                running_corrects_draw += torch.sum(preds_draw == draw_labels.data)

            if phase == 'train':
                scheduler_image.step()
                scheduler_draw.step()


            epoch_loss_image = running_loss_image / dataset_sizes[phase]
            epoch_loss_draw = running_loss_draw / dataset_sizes[phase]

            epoch_dist_loss = running_dist_loss / dataset_sizes[phase]

            epoch_cross_entropy_loss_image = running_cross_entropy_loss_image / dataset_sizes[phase]
            epoch_cross_entropy_loss_draw = running_cross_entropy_loss_draw / dataset_sizes[phase]

            epoch_acc_image = 100 * running_corrects_image.float() / dataset_sizes[phase]
            epoch_acc_draw = 100 * running_corrects_draw.float() / dataset_sizes[phase]

            if phase == 'train':
                train_losses_image.append(epoch_loss_image)
                train_losses_draw.append(epoch_loss_draw)

                train_accuracy_image.append(epoch_acc_image.item())
                train_accuracy_draw.append(epoch_acc_draw.item())

                print('{} image Loss: {:.4f}, draw loss: {:.4f}, dist_loss: {:.4f}, cross_entropy_loss_image: {:.4f}, cross_entropy_loss_draw: {:.4f}, '
                      'acc_image: {:.4f}, acc_draw: {:.4f}'.format(
                    phase, epoch_loss_image, epoch_loss_draw, epoch_dist_loss, epoch_cross_entropy_loss_image, epoch_cross_entropy_loss_draw,
                    epoch_acc_image, epoch_acc_draw))
            if phase == 'test':
                total_loss_image = running_loss_image / dataset_sizes['test']
                total_loss_draw = running_loss_draw / dataset_sizes['test']

                total_dist_loss = running_dist_loss / dataset_sizes['test']

                total_cross_entropy_loss_image = running_cross_entropy_loss_image / dataset_sizes['test']
                total_cross_entropy_loss_draw = running_cross_entropy_loss_draw / dataset_sizes['test']

                acc_image = 100 * running_corrects_image.float() / dataset_sizes['test']
                acc_draw = 100 * running_corrects_draw.float() / dataset_sizes['test']

                if epoch == 0:
                    best_acc_image = acc_image
                    best_acc_draw = acc_draw
                if acc_image >= best_acc_image:
                    best_acc_image = acc_image
                    print("new best image acc {}".format(best_acc_image))
                    filename = os.path.join(image_checkpoint_dir, exp + '_best_model' + '.pth.tar')
                    torch.save(image_model.state_dict(), filename)
                    best_model_wts_image = copy.deepcopy(image_model.state_dict())
                print("Image acc is {} loss is {} dist loss is {} cross entropy loss is {}".format(acc_image, total_loss_image,
                                                                                             total_dist_loss, total_cross_entropy_loss_image))
                if acc_draw >= best_acc_draw:
                    best_acc_draw = acc_draw
                    print("new best draw acc {}".format(best_acc_draw))
                    filename = os.path.join(draw_checkpoint_dir, exp + '_best_model' + '.pth.tar')
                    torch.save(drawing_model.state_dict(), filename)
                    best_model_wts_draw = copy.deepcopy(drawing_model.state_dict())
                print("Draw acc is {} loss is {} dist loss is {} cross entropy loss is {}".format(acc_draw, total_loss_draw,
                                                                                             total_dist_loss, total_cross_entropy_loss_draw))


    filename = os.path.join(image_checkpoint_dir, exp + '_best_model' + '.pth.tar')
    checkpoint = torch.load(filename, map_location="cpu")
    # self.model.load_state_dict(checkpoint["state_dict"], strict=False)
    image_model.load_state_dict(checkpoint)

    filename = os.path.join(draw_checkpoint_dir, exp + '_best_model' + '.pth.tar')
    checkpoint = torch.load(filename, map_location="cpu")
    # self.model.load_state_dict(checkpoint["state_dict"], strict=False)
    drawing_model.load_state_dict(checkpoint)

    return image_model, drawing_model