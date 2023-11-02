from PIL import Image
import requests
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T


class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))  

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)  # (B, 3, H, W) -> (B, 64, H/2, W/2)
        x = self.backbone.bn1(x)  # (B, 64, H/2, W/2)
        x = self.backbone.relu(x)  # (B, 64, H/2, W/2)
        x = self.backbone.maxpool(x)  # (B, 64, H/4, W/4)

        x = self.backbone.layer1(x)  # (B, 64, H/4, W/4) -> (B, 256, H/4, W/4)
        x = self.backbone.layer2(x)  # (B, 256, H/4, W/4) -> (B, 512, H/8, W/8)
        x = self.backbone.layer3(x)  # (B, 512, H/8, W/8) -> (B, 1024, H/16, W/16)
        x = self.backbone.layer4(x)  # (B, 1024, H/16, W/16) -> (B, 2048, H/32, W/32)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)  # (B, 2048, H/32, W/32) -> (B, 256, H/32, W/32)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),  # (H, W, hidden_dim//2)
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),  # (H, W, hidden_dim//2)
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # (H*W, 1, hidden_dim)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1)).transpose(0, 1)
        # (H*W, 1, hidden_dim), (100, 1, hidden_dim) -> (100, 1, hidden_dim) -> (1, 100, hidden_dim)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {
            'pred_logits': self.linear_class(h),  # (1, 100, num_classes+1)
            'pred_boxes': self.linear_bbox(h).sigmoid(),  # (1, 100, 4)
            }


detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', 
    check_hash=True,
)
detr.load_state_dict(state_dict)
detr.eval()

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # (100, num_classes) 
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)  # Resize to (800, 1066)

scores, boxes = detect(im, detr, transform)
print(scores, boxes)