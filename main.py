from __future__ import print_function

import os

import cv2
import matplotlib.pyplot as plt

from imports import *
from data_split import *
from visualize import *
from train_test import *
from model import *
from config import *
from myensemble import *
from train_image_to_draw import *

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

class MyDataset(Dataset):
    def __init__(self, draw_datasets, image_datasets):
        self.draw_datasets = draw_datasets # datasets should be sorted!
        self.image_datasets = image_datasets

    def __getitem__(self, index):
        x1 = self.draw_datasets[index]
        x2 = self.image_datasets[index]
        return x1, x2

    def __len__(self):
        return len(self.draw_datasets) # assuming both datasets have same length


def data_transforms(size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def data_loader(data_dir, batch_size, data_transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names


def data_loader_both(draw_dir, image_dir, batch_size, data_transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(image_dir, x),
                                                   data_transforms[x])
                           for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    train_image_dataset = datasets.ImageFolder(os.path.join(image_dir, 'train'), data_transforms['train'])
    train_draw_dataset = datasets.ImageFolder(os.path.join(draw_dir, 'train'), data_transforms['train'])

    test_image_dataset = datasets.ImageFolder(os.path.join(image_dir, 'test'), data_transforms['test'])
    test_draw_dataset = datasets.ImageFolder(os.path.join(draw_dir, 'test'), data_transforms['test'])

    dataset_train = MyDataset(train_draw_dataset, train_image_dataset)
    dataset_test = MyDataset(test_draw_dataset, test_image_dataset)

    dataloaders_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloaders_train, dataloaders_test, dataset_sizes, class_names


def print_torch():
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

def attention_map(gener):
    if gener==True:
        model_name_dir = "deit_vit_model_with_generated"
    else:
        model_name_dir = "deit_vit_model_without_generated"
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    pretrained_model = ViTModel.from_pretrained(model_name_dir, add_pooling_layer=False, output_attentions=True)
    animal_dir = os.path.join("glyptic_images/with_generated_data/test/", "animal")
    not_animal_dir = os.path.join("glyptic_images/with_generated_data/test/", "not_animal")
    # url1 = 'https://i.natgeofe.com/k/d21630fa-3ab9-4e37-adea-c503629e49d4/great_white_smile.jpg'
    for current_dir in [animal_dir, not_animal_dir]:
        i = 0
        for image_path in os.listdir(current_dir):
            i +=1
            if i > 30:
                break
            image_path = os.path.join(current_dir, image_path)
            image = Image.open(image_path).convert("RGB")
            # image = image.resize((224, 224))
            # print(image.shape)
            # image = Image.open(requests.get(url1, stream=True).raw)
            # print(image.size)
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = pretrained_model(**inputs)
            attentions = outputs.attentions

            att_mat = torch.stack(attentions).squeeze(1)

            # attention 평균
            att_mat = reduce(att_mat, 'b h len1 len2 -> b len1 len2', 'mean')
            im = np.array(image)

            residual_att = torch.eye(att_mat.size(1))
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size())
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

            # Attention from the output token to the input space.
            v = joint_attentions[-1]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[0]))[..., np.newaxis]
            result = (mask * im).astype("uint8")

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))

            # img_show = Image.fromarray(result, 'RGB')
            # img_show.show()

            # plt.imshow(result, interpolation='nearest')
            # plt.show()
            if current_dir==animal_dir:
                if gener==True:
                    ax1.set_title('Original animal trained with generated data')
                    ax2.set_title('Attention Mask')
                    ax3.set_title('Attention Map')
                    ax1.imshow(im, interpolation='nearest')
                    ax2.imshow(mask.squeeze(), interpolation='nearest')
                    ax3.imshow(result, interpolation='nearest')
                    plt.show()
                else:
                    ax1.set_title('Original animal trained without generated data')
                    ax2.set_title('Attention Mask')
                    ax3.set_title('Attention Map')
                    ax1.imshow(im, interpolation='nearest')
                    ax2.imshow(mask.squeeze(), interpolation='nearest')
                    ax3.imshow(result, interpolation='nearest')
                    plt.show()
            else:
                if gener==True:
                    ax1.set_title('Original not animal trained with generated data')
                    ax2.set_title('Attention Mask')
                    ax3.set_title('Attention Map')
                    ax1.imshow(im, interpolation='nearest')
                    ax2.imshow(mask.squeeze(), interpolation='nearest')
                    ax3.imshow(result, interpolation='nearest')
                    plt.show()
                else:
                    ax1.set_title('Original not animal trained without generated data')
                    ax2.set_title('Attention Mask')
                    ax3.set_title('Attention Map')
                    ax1.imshow(im, interpolation='nearest')
                    ax2.imshow(mask.squeeze(), interpolation='nearest')
                    ax3.imshow(result, interpolation='nearest')
                    plt.show()

def visualize_filters(model, gener, deivce):
    model.cpu()
    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")
    animal_dir = os.path.join("glyptic_images/with_generated_data/test/", "animal")
    not_animal_dir = os.path.join("glyptic_images/with_generated_data/test/", "not_animal")
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
    for current_dir in [animal_dir, not_animal_dir]:
        i = 0
        for image_path in os.listdir(current_dir):
            i +=1
            if i > 2:
                break
            image_path = os.path.join(current_dir, image_path)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)
            # apply the transforms
            img = transform(img)
            # unsqueeze to add a batch dimension
            img = img.unsqueeze(0)
            # pass the image through all the layers
            results = [conv_layers[0](img)]
            for j in range(1, len(conv_layers)):
                # pass the result from the last layer to the next layer
                results.append(conv_layers[j](results[-1]))

            # make a copy of the `results`
            outputs = results
            # visualize 64 features from each layer
            # (although there are more feature maps in the upper layers)
            for num_layer in range(len(outputs)):
                plt.figure(figsize=(30, 30))
                layer_viz = outputs[num_layer][0, :, :, :]
                layer_viz = layer_viz.data
                print(layer_viz.size())
                for i, filter in enumerate(layer_viz):
                    if i == 64:  # we will visualize only 8x8 blocks from each layer
                        break
                    plt.subplot(8, 8, i + 1)
                    plt.imshow(filter, cmap='gray')
                    plt.axis("off")
                # print(f"Saving layer {num_layer} feature maps...")
                # plt.savefig(f"../outputs/layer_{num_layer}.png")
                plt.show()
                plt.close()

    # take a look at the conv layers and the respective weights
    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        if(gener==True):
            plt.title("With generated data")
        else:
            plt.title("Without generated data")
        plt.subplot(8, 8, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        # plt.savefig('../outputs/filter.png')
    plt.show()


if __name__ == '__main__':
    print("arch recognition")
    # animal_generator()
    print_torch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = config(vit=0, gamma=0.95, step_size=6, num_epochs=0, batch_size=8,
                    train_resnet=True, train_deit=False, num_classes=10, image_size=224, lr=0.00005)

    finetune_draws_on_photos = False
    photo_as_draw = False
    eval_only = True
    num_of_exps_to_eval = 10

    # config = config(vit=0, gamma=0.8, step_size=8, num_epochs=0, batch_size=32,
    #                 train_resnet=False, train_deit=True, num_classes=10, image_size=224, lr=0.00003)

    # data_split()

    # plt.ion()  # interactive mode

    data_transforms = data_transforms(config.image_size)
    batch_size = config.batch_size
    photo_confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
    draw_confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
    photo_to_draw_confusion_matrix = torch.zeros(config.num_classes, config.num_classes)

    acc_photo_total, acc_draw_total, acc_photo_to_draw_total = 0.0, 0.0, 0.0
    # for exp in ['experiment_11', 'experiment_12', 'experiment_13',
    #             'experiment_14', 'experiment_15', 'experiment_16', 'experiment_17', 'experiment_18', 'experiment_19']:
    for exp in ['experiment_0', 'experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5', 'experiment_6',
                'experiment_7', 'experiment_8', 'experiment_9']:
        for data in ["photo", "draw", "photo_to_draw"]:
            project_name = "classification_force_photo_to_draw"
            run_wandb = wandb.init(project=project_name, entity="offry", reinit=True)
            experiment_name = "resnet152_10_classes"
            run_wandb.name = experiment_name
            run_wandb.save()
            run_wandb.config.gamma, run_wandb.config.step_size, run_wandb.config.lr = config.gamma, config.step_size, config.lr

        # for data in ["photo_to_draw"]:
            if data=="photo_to_draw":
                train_image_to_draw_flag = True
            else:
                train_image_to_draw_flag = False
            if data=="photo":
                if photo_as_draw:
                    data_dir = 'images_final/labeled_photo_as_drawing/supervised_experiments/' + exp
                    checkpoint_dir = os.path.join(os.getcwd(),
                                                  "checkpoints/checkpoints_" + data + "_as_drawing" + "_" + exp + "_num_classes_" + str(
                                                      config.num_classes))
                    if not os.path.isdir(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                else:
                    data_dir = 'images_final/labeled_photo/supervised_experiments/' + exp
                    checkpoint_dir = os.path.join(os.getcwd(),
                                                  "checkpoints/checkpoints_" + data + "_" + exp + "_num_classes_" + str(
                                                      config.num_classes))
                    if not os.path.isdir(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
            elif data=="photo_to_draw":
                image_dir = 'images_final/labeled_photo/supervised_experiments/' + exp
                checkpoint_dir = os.path.join(os.getcwd(),
                                              "checkpoints/checkpoints_image_to_draw_" + exp + "_num_classes_" + str(
                                                  config.num_classes))
                if not os.path.isdir(checkpoint_dir):
                    os.mkdir(checkpoint_dir)

                draw_dir = 'images_final/labeled_drawing/supervised_experiments/' + exp
                draw_checkpoint_dir = os.path.join(os.getcwd(),
                                                   "checkpoints/checkpoints_draw_" + exp + "_num_classes_" + str(
                                                       config.num_classes))
                if not os.path.isdir(draw_checkpoint_dir):
                    os.mkdir(draw_checkpoint_dir)
            else:
                data_dir = 'images_final/labeled_drawing/supervised_experiments/' + exp
                checkpoint_dir = os.path.join(os.getcwd(),
                                              "checkpoints/checkpoints_" + data + "_" + exp + "_num_classes_" + str(
                                                  config.num_classes))
                if not os.path.isdir(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
            if train_image_to_draw_flag:
                dataloaders_train, dataloaders_test, dataset_sizes, class_names = data_loader_both(draw_dir, image_dir, batch_size, data_transforms)
            else:
                dataloaders, dataset_sizes, class_names = data_loader(data_dir, batch_size, data_transforms)
            print(dataset_sizes['train'])
            print(dataset_sizes['test'])

            # Observe that all parameters are being optimized
            criterion = nn.CrossEntropyLoss()
            if config.train_deit:

                # if config.vit:
                # model_deit = ViTModel.from_pretrained('google/vit-base-patch16-224',
                #                                            add_pooling_layer=False,
                #                                            output_attentions=True)
                model_pre_deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                       add_pooling_layer=False,
                                                                          output_attentions=True)
                
                model_deit = model(model_pre_deit, 768, config.num_classes)
                model_deit.to(device)
                optimizer_model_deit = optim.Adam(model_deit.parameters(), config.lr)
                # Decay LR by a factor of 0.1 every 3 epochs
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_model_deit, step_size=config.step_size, gamma=config.gamma)

                train_test_class_deit = train_test(model_deit, criterion, optimizer_model_deit, exp_lr_scheduler,
                                                     config.num_epochs,
                                                     dataloaders, device, dataset_sizes,
                                                     exp, class_names, checkpoint_dir, model_name='deit_vit')

                model_test = train_test_class_deit.train_model()
                del model_deit

            if config.train_resnet:
                model_resnet = resnet(pretrained=True, num_classes=config.num_classes).build_resnet().to(device)
                optimizer_model_resnet = optim.Adam(model_resnet.parameters(), config.lr)
                # Decay LR by a factor of 0.1 every 3 epochs
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_model_resnet, step_size=config.step_size, gamma=config.gamma)

                if finetune_draws_on_photos:
                    load_checkpoint_dir = os.path.join(os.getcwd(),
                                                  "checkpoints/checkpoints_" + "draw" + "_" + exp + "_num_classes_" + str(
                                                      config.num_classes))
                    filename = os.path.join(load_checkpoint_dir, exp + '_best_model' + '.pth.tar')
                    checkpoint = torch.load(filename, map_location="cpu")
                    model_resnet.load_state_dict(checkpoint)
                if eval_only:
                    if train_image_to_draw_flag:
                        load_checkpoint_dir = os.path.join(os.getcwd(),
                                                           "checkpoints/checkpoints_image_to_draw_" + exp + "_num_classes_" + str(
                                                               config.num_classes))

                    else:
                        load_checkpoint_dir = os.path.join(os.getcwd(),
                                                           "checkpoints/checkpoints_" + data + "_" + exp + "_num_classes_" + str(
                                                               config.num_classes))
                    filename = os.path.join(load_checkpoint_dir, exp + '_best_model' + '.pth.tar')
                    checkpoint = torch.load(filename, map_location="cpu")
                    model_resnet.load_state_dict(checkpoint)

                if train_image_to_draw_flag:
                    if config.num_epochs > 0:
                        model_test = train_image_network_as_draw(config.num_epochs, "resnet_model", model_resnet,
                                                                 device, dataloaders_train, dataloaders_test,
                                                                 optimizer_model_resnet, criterion, exp_lr_scheduler,
                                                                 dataset_sizes, exp, checkpoint_dir,
                                                                 draw_checkpoint_dir)
                    else:
                        model_test = model_resnet
                else:
                    train_test_class_resnet = train_test(model_resnet, criterion, optimizer_model_resnet, exp_lr_scheduler,
                                                         config.num_epochs,
                                                         dataloaders, device, dataset_sizes,
                                                         exp, class_names, checkpoint_dir, model_name='resnet_model')

                    if config.num_epochs > 0:
                        model_test = train_test_class_resnet.train_model()
                    else:
                        model_test = model_resnet
            model_test.eval()

            # Iterate over data.
            incorrect_examples = []
            original_labels = []
            pred_labels = []
            real_classes_num = 0
            real_class_names = []
            running_corrects = 0
            for sub_class in class_names:
                if "_" in sub_class:
                    if sub_class.split("_")[0] in real_class_names:
                        continue
                    else:
                        real_class_names.append(sub_class.split("_")[0])
                        real_classes_num+=1
                else:
                    if sub_class.split("_")[0] in real_class_names:
                        continue
                    else:
                        real_class_names.append(sub_class)
                        real_classes_num += 1
            confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
            with torch.no_grad():
                if train_image_to_draw_flag:
                    for draw_data, image_data in dataloaders_test:
                        draw_inputs, draw_labels = draw_data
                        image_inputs, image_labels = image_data
    
                        draw_inputs = draw_inputs.to(device)
                        # draw_labels = draw_labels.to(device)
                        image_inputs = image_inputs.to(device)
                        image_labels = image_labels.to(device)
    
                        outputs = model_test(image_inputs)
    
                        _, preds = torch.max(outputs, 1)
                        new_preds = preds
                        for i in range(len(preds)):
                            pred = preds[i]
                            pred_name = class_names[preds[i]]
                            label_name = class_names[image_labels[i].item()]
                            if "_" in label_name:
                                label_name = label_name.split("_")[0]
                            if "_" in pred_name:
                                pred_name = pred_name.split("_")[0]
                            if label_name==pred_name:
                                new_preds[i]=image_labels[i]
                        running_corrects += torch.sum(new_preds == image_labels.data)
                        idxs_mask = (preds != image_labels).view(-1)
                        incorrect_examples.append(image_inputs[idxs_mask].cpu())
                        original_labels.append(image_labels[idxs_mask].cpu())
                        pred_labels.append(preds[idxs_mask].cpu())
                        for t, p in zip(image_labels.view(-1), preds.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                else:
                    for inputs, labels in dataloaders['test']:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model_test(inputs)

                        _, preds = torch.max(outputs, 1)
                        new_preds = preds
                        for i in range(len(preds)):
                            pred = preds[i]
                            pred_name = class_names[preds[i]]
                            label_name = class_names[labels[i].item()]
                            if "_" in label_name:
                                label_name = label_name.split("_")[0]
                            if "_" in pred_name:
                                pred_name = pred_name.split("_")[0]
                            if label_name == pred_name:
                                new_preds[i] = labels[i]
                        running_corrects += torch.sum(new_preds == labels.data)
                        idxs_mask = (preds != labels).view(-1)
                        incorrect_examples.append(inputs[idxs_mask].cpu())
                        original_labels.append(labels[idxs_mask].cpu())
                        pred_labels.append(preds[idxs_mask].cpu())
                        for t, p in zip(labels.view(-1), preds.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1

                for i in range(len(confusion_matrix)):
                    confusion_matrix[i] = confusion_matrix[i] / torch.sum(confusion_matrix[i], dim=0)
                if data == "photo":
                    photo_confusion_matrix = torch.add(photo_confusion_matrix, confusion_matrix)
                elif data == "photo_to_draw":
                    photo_to_draw_confusion_matrix = torch.add(photo_to_draw_confusion_matrix, confusion_matrix)
                else:
                    draw_confusion_matrix = torch.add(draw_confusion_matrix, confusion_matrix)
                fig = plt.figure(figsize=(22, 18))
                import pandas as pd
                import seaborn as sns
                plt.interactive(True)
                df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(float)
                heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
                acc = 100 * running_corrects.float() / dataset_sizes['test']
                if data=='photo':
                    acc_photo_total+=acc
                elif data=="photo_to_draw":
                    acc_photo_to_draw_total+=acc
                else:
                    acc_draw_total+=acc
                heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
                heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=15)
                if train_image_to_draw_flag:
                    plt.title("{}, {}, acc is {}".format(exp, "photo to draw", acc), fontsize=15)
                else:
                    plt.title("{}, {}, acc is {}".format(exp, data, acc), fontsize=15)
                plt.ylabel('True label', fontsize=15)
                plt.xlabel('Predicted label', fontsize=15)
                plt.show()
                if train_image_to_draw_flag:
                    # fig.savefig("figures/" + exp + "_photo_to_draw" + ".png")
                    run_wandb.log({exp + "_photo_to_draw": fig})
                else:
                    # fig.savefig("figures/" + exp + "_" + data + ".png")
                    run_wandb.log({exp + "_" + data: fig})
                plt.close
                run_wandb.log({"accuracy_" + data + "_" + exp: acc})
                print("num wrong: {}".format(dataset_sizes['test'] - running_corrects))
                print("{}, {}, acc is {}".format(exp, data, acc))
                num_of_exps_to_eval_str = str(num_of_exps_to_eval-1)
                if num_of_exps_to_eval_str in exp:
                    if data == "photo":
                        fig = plt.figure(figsize=(22, 18))
                        acc_photo_total = acc_photo_total / num_of_exps_to_eval
                        photo_confusion_matrix = torch.div(photo_confusion_matrix, num_of_exps_to_eval)
                        df_cm = pd.DataFrame(photo_confusion_matrix, index=class_names, columns=class_names).astype(float)
                        heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
                        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',
                                                     fontsize=15)
                        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right',
                                                     fontsize=15)
                        if train_image_to_draw_flag:
                            plt.title("Total {}, acc is {}".format("photo to draw", acc_photo_total), fontsize=15)
                        else:
                            plt.title("Total {}, acc is {}".format(data, acc_photo_total), fontsize=15)
                        plt.ylabel('True label', fontsize=15)
                        plt.xlabel('Predicted label', fontsize=15)
                        plt.show()
                        # fig.savefig("figures/total_" + data + ".png")
                        run_wandb.log({"total_" + exp + "_" + data: fig})
                        run_wandb.log({"accuracy_photo_total": acc_photo_total})
                    elif data=="photo_to_draw":
                        fig = plt.figure(figsize=(22, 18))
                        acc_photo_to_draw_total = acc_photo_to_draw_total / num_of_exps_to_eval
                        photo_to_draw_confusion_matrix = torch.div(photo_to_draw_confusion_matrix, num_of_exps_to_eval)
                        df_cm = pd.DataFrame(photo_to_draw_confusion_matrix, index=class_names, columns=class_names).astype(
                            float)
                        heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
                        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',
                                                     fontsize=15)
                        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right',
                                                     fontsize=15)
                        plt.title("Total {}, acc is {}".format(data, acc_photo_to_draw_total), fontsize=15)
                        plt.ylabel('True label', fontsize=15)
                        plt.xlabel('Predicted label', fontsize=15)
                        plt.show()
                        # fig.savefig("figures/total_" + "photo_to_draw" + ".png")
                        run_wandb.log({"total_" + exp + "_" + "photo_to_draw": fig})
                        run_wandb.log({"accuracy_photo_to_draw_total": acc_photo_to_draw_total})
                    else:
                        fig = plt.figure(figsize=(22, 18))
                        acc_draw_total = acc_draw_total / num_of_exps_to_eval
                        draw_confusion_matrix = torch.div(draw_confusion_matrix, num_of_exps_to_eval)
                        df_cm = pd.DataFrame(draw_confusion_matrix, index=class_names, columns=class_names).astype(float)
                        heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
                        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',
                                                     fontsize=15)
                        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right',
                                                     fontsize=15)
                        plt.title("Total {}, acc is {}".format(data, acc_draw_total), fontsize=15)
                        plt.ylabel('True label', fontsize=15)
                        plt.xlabel('Predicted label', fontsize=15)
                        plt.show()
                        # fig.savefig("figures/total_" + data + ".png")
                        run_wandb.log({"total_" + exp + "_" + data: fig})
                        run_wandb.log({"accuracy_draw_total": acc_draw_total})
                # for j, item in enumerate(incorrect_examples):
                #     for i, image in enumerate(item):
                #         invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                #                                                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                #                                        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                #                                                             std=[1., 1., 1.]),
                #                                        ])
                #
                #         inv_tensor = invTrans(image)
                #         plt.figure()
                #         plt.title("label: {} pred label: {}".format(class_names[int(original_labels[j][i])],
                #                                                     class_names[int(pred_labels[j][i])]))
                #         plt.imshow(inv_tensor.permute(1, 2, 0))
                #         plt.show()
                #         run_wandb.log({str(i): fig})
                run_wandb.finish()
                del model_test