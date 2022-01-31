from __future__ import print_function

from imports import *
plt.ion()

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # plt.imshow(inp)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    return inp

class visualizer:

    def __init__(self, model, device, dataloaders, class_names):
        self.model = model
        self.device = device
        self.dataloaders = dataloaders
        self.class_names = class_names


    def visualize_model_predictions(self):
        model = self.model
        class_names = self.class_names
        dataloaders = self.dataloaders
        device = self.device

        was_training = model.training
        num_images = 6
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['test']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    out = imshow(inputs.cpu().data[j])
                    plt.imshow(out)
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)