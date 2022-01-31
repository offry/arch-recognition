from imports import *

# class model(nn.Module):
#     def __init__(self, pretrained=False, num_classes=2, model_name='vit_deit_base_patch16_224'):
#         super().__init__()
#         self.model = timm.create_model(model_name, pretrained=pretrained)
#         ### vit
#         num_features = self.model.head.in_features
#         self.model.head = nn.Linear(num_features, num_classes)
#
#         '''
#         self.model.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             #nn.Linear(num_features, hidden_size,bias=True), nn.ELU(),
#             nn.Linear(num_features, num_classes, bias=True)
#         )
#         '''
#
#     def forward(self, x):
#         x = self.model(x)
#         return x

class model(nn.Module):
    def __init__(self, pretrained_vit_model, d_model, classes):
        super().__init__()
        self.pretrained_vit_model = pretrained_vit_model
        self.classifier = nn.Linear(d_model, classes)

    def forward(self, x):
        x = self.pretrained_vit_model(x)
        attentions = x.attentions
        output = self.classifier(x.last_hidden_state[:, 0, :])  # cls tokken

        return output