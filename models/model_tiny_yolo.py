import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def forward(self, x, targets=None, img_dim=None):
        num_samples = x.size(0)
        grid_size = x.size(2)
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        output = torch.cat(
            (
                x.view(num_samples, -1, 1),
                y.view(num_samples, -1, 1),
                w.view(num_samples, -1, 1),
                h.view(num_samples, -1, 1),
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        return output

class TinyYOLOv3(nn.Module):
    def __init__(self, num_classes=80, img_dim=416):
        super(TinyYOLOv3, self).__init__()
        self.module_list = nn.ModuleList([
            self._make_layer(3, 16, 3, 1),
            nn.MaxPool2d(2, 2),
            self._make_layer(16, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            self._make_layer(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            self._make_layer(64, 128, 3, 1),
            nn.MaxPool2d(2, 2),
            self._make_layer(128, 256, 3, 1),
            nn.MaxPool2d(2, 2),
            self._make_layer(256, 512, 3, 1),
            nn.MaxPool2d(2, 1, 1),
            self._make_layer(512, 1024, 3, 1),
            self._make_layer(1024, 256, 1, 1),
            self._make_layer(256, 512, 3, 1),
            nn.Conv2d(512, (4 + 1 + num_classes) * 3, 1, 1, 0)
        ])
        self.yolo_layer = YOLOLayer(
            anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)],
            num_classes=num_classes,
            img_dim=img_dim
        )
        self.img_dim = img_dim

    def _make_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, targets=None):
        layer_outputs, yolo_outputs = [], []
        for i, module in enumerate(self.module_list):
            x = module(x)
            if i == 13:
                yolo_outputs.append(self.yolo_layer(x))
        return yolo_outputs

def TinyYOLOv3Model(num_classes=80, img_dim=416):
    return TinyYOLOv3(num_classes, img_dim)
