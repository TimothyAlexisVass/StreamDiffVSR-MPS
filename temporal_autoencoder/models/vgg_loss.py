import torch.nn as nn

class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg, feature_layers=[2, 7, 12, 21]):  
        """
        Extract different layers from VGG-19 to compute perceptual loss
        Default extraction:
        - relu1_2 (layer 2)
        - relu2_2 (layer 7)
        - relu3_4 (layer 12)
        - relu4_4 (layer 21)
        """
        super(VGGPerceptualLoss, self).__init__()
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(feature_layers)+1])
        self.feature_layers = feature_layers
        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # Freeze VGG to avoid affecting training
    
    def forward(self, pred, target):
        loss = 0
        for i, layer in enumerate(self.vgg_layers):
            pred = layer(pred)
            target = layer(target)
            if i in self.feature_layers:
                loss += nn.functional.l1_loss(pred, target)  # L1 loss is more suitable for high-level features
        return loss
