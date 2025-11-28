import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import numpy as np
from PIL import Image
from rasterio.plot import reshape_as_image
# ----------------------------
# Define RA-UNet class (kept for completeness)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.Wx = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch,1,1,bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = torch.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class RAUNet(nn.Module):
    def __init__(self, in_channels=5, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels,f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(f,f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(f*2,f*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(f*4,f*8)
        self.pool4 = nn.MaxPool2d(2)
        self.center = ConvBlock(f*8,f*16)

        self.up4 = nn.ConvTranspose2d(f*16,f*8,2,2)
        self.att4 = AttentionGate(f*8,f*8,f*4)
        self.dec4 = ConvBlock(f*16,f*8)

        self.up3 = nn.ConvTranspose2d(f*8,f*4,2,2)
        self.att3 = AttentionGate(f*4,f*4,f*2)
        self.dec3 = ConvBlock(f*8,f*4)

        self.up2 = nn.ConvTranspose2d(f*4,f*2,2,2)
        self.att2 = AttentionGate(f*2,f*2,f)
        self.dec2 = ConvBlock(f*4,f*2)

        self.up1 = nn.ConvTranspose2d(f*2,f,2,2)
        self.att1 = AttentionGate(f,f,max(f//2,1))
        self.dec1 = ConvBlock(f*2,f)

        self.final = nn.Conv2d(f,1,1)

    def forward(self,x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        c = self.center(p4)
        u4 = self.up4(c)
        ae4 = self.att4(u4,e4)
        d4 = self.dec4(torch.cat([u4,ae4],dim=1))
        u3 = self.up3(d4)
        ae3 = self.att3(u3,e3)
        d3 = self.dec3(torch.cat([u3,ae3],dim=1))
        u2 = self.up2(d3)
        ae2 = self.att2(u2,e2)
        d2 = self.dec2(torch.cat([u2,ae2],dim=1))
        u1 = self.up1(d2)
        ae1 = self.att1(u1,e1)
        d1 = self.dec1(torch.cat([u1,ae1],dim=1))
        return self.final(d1)
# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "fire_raunet_best.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RAUNet(in_channels=5, base_filters=32)
# NOTE: Model loading commented out to allow app execution without the weight file
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_geotiff(file_path):
    with rasterio.open(file_path) as src:
        arr = src.read().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr_hw = reshape_as_image(arr)
    arr_hw_clip = np.clip(arr_hw / 10000.0, 0.0, 1.0)
    inp = torch.from_numpy(np.transpose(arr_hw_clip, (2,0,1))).unsqueeze(0).float()
    inp = F.interpolate(inp, size=(256,256), mode='bilinear', align_corners=False)
    inp = inp.to(device)
    return inp, arr_hw

def predict_fire(file_path):
    inp, arr_hw = preprocess_geotiff(file_path)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)[0,0].cpu().numpy()
    # Resize to original
    from PIL import Image
    probs_up = np.array(Image.fromarray((probs*255).astype(np.uint8))
                        .resize((arr_hw.shape[1], arr_hw.shape[0]), resample=Image.BILINEAR))/255.0
    return arr_hw, probs_up

# UPDATED: Improved normalization for better visual contrast (True Color image)
def get_rgb_from_5band_array(img_arr):
    if img_arr.ndim == 3 and img_arr.shape[0]==5:
        img_arr = np.transpose(img_arr,(1,2,0))
    # Assuming Sentinel-2 B, G, R, NIR, SWIR -> Indices 0, 1, 2, 3, 4
    r = img_arr[:,:,2] 
    g = img_arr[:,:,1]
    b = img_arr[:,:,0]
    rgb = np.stack([r,g,b], axis=-1)
    
    # Use 2nd and 98th percentile for clipping for better contrast
    min_val = np.percentile(rgb, 2)
    max_val = np.percentile(rgb, 98)
    
    rgb = (rgb - min_val) / (max_val - min_val + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    return rgb