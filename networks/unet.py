import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class CS(nn.Module):
    def __init__(self, features, WH, r, L=32, num_branches=3):
        super(CS, self).__init__()
        self.num_branches = num_branches
        d = max(int(features / r), L)

        self.gap = nn.AvgPool2d(int(WH))  # Global Average Pooling
        self.fc = nn.Linear(features * 3, d)  # Use all three branches for attention base
        self.fcs = nn.ModuleList([nn.Linear(d, features) for _ in range(num_branches)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Unpack branches: assume order is (l_freq, h_freq, of)
        x_low, x_high, x_flow = x        # Use all three branches to compute attention
        x_all = torch.cat([x_low, x_high, x_flow], dim=1)  # Concatenate all branches
        fea_s = self.gap(x_all).squeeze_()  # Global descriptor
        fea_z = self.fc(fea_s.cpu())  # Bottleneck

        # Compute attention weights per branch
        attention_vec = torch.cat([
            fc(fea_z).unsqueeze(1) for fc in self.fcs
        ], dim=1)
        attention_vec = self.softmax(attention_vec)
        attention_vec = attention_vec.unsqueeze(-1).unsqueeze(-1)  # For broadcast

        device = x_low.device
        attention_vec = attention_vec.transpose(0, 1).to(device)

        # Apply attention to each branch
        out_low = x_low * attention_vec[0]
        out_high = x_high * attention_vec[1]
        out_flow = x_flow * attention_vec[2]

        return (out_low, out_high, out_flow)

class UNet(nn.Module):
    def __init__(self, 
                 l_freq_channels=12,    # Low frequency input channels
                 h_freq_channels=12,   # High frequency input channels  
                 of_channels=6, # Optical flow input channels
                 output_channels=3):      # Output channels (RGB)
        super(UNet, self).__init__()
        
        # ========== LOW FREQUENCY BRANCH ==========
        self.inc_l_freq = inconv(l_freq_channels, 64)
        self.down1_l_freq = down(64, 128)
        self.down2_l_freq = down(128, 256)
        self.down3_l_freq = down(256, 512)
        
        # ========== HIGH FREQUENCY BRANCH ==========
        self.inc_h_freq = inconv(h_freq_channels, 64)
        self.down1_h_freq = down(64, 128)
        self.down2_h_freq = down(128, 256)
        self.down3_h_freq = down(256, 512)
        
        # ========== OPTICAL FLOW BRANCH ==========
        self.inc_of = inconv(of_channels, 64)
        self.down1_of = down(64, 128)
        self.down2_of = down(128, 256)
        self.down3_of = down(256, 512)

        # ========== CHANNEL SHUFFLE ATTENTION ==========
        self.cs1 = CS(features=64, WH=256, r=16)   # For level 1 (64 channels, 256x256)
        self.cs2 = CS(features=128, WH=128, r=16)  # For level 2 (128 channels, 128x128)
        self.cs3 = CS(features=256, WH=64, r=16)   # For level 3 (256 channels, 64x64)
        self.cs4 = CS(features=512, WH=32, r=16)   # For level 4 (512 channels, 32x32)
        
        # ========== FEATURE FUSION ==========
        # Fusion layers
        self.fusion_conv = double_conv(512 * 3, 512)  # Combine all 3 branches
        self.se = SELayer(512)  # Channel attention
        
        # ========== DECODER (Single Path) ==========
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        
    def forward(self, l_freq, h_freq, of):
        '''
        Args:
            l_freq: Low frequency components [B, l_freq_channels, H, W]
            h_freq: High frequency components [B, h_freq_channels, H, W]
            of: Optical flow between frames [B, of_channels, H, W]
        
        Returns:
            reconstructed: Reconstructed frames [B, output_channels, H, W]
            features: Dictionary of intermediate features for loss computation
        '''        

        l_freq1 = self.inc_l_freq(l_freq)         
        h_freq1 = self.inc_h_freq(h_freq)     
        of1 = self.inc_of(of)     
        l_freq1, h_freq1, of1 = self.cs1((l_freq1, h_freq1, of1))


        l_freq2 = self.down1_l_freq(l_freq1)       
        h_freq2 = self.down1_h_freq(h_freq1)    
        of2 = self.down1_of(of1) 
        l_freq2, h_freq2, of2 = self.cs2((l_freq2, h_freq2, of2))
        

        l_freq3 = self.down2_l_freq(l_freq2)       
        h_freq3 = self.down2_h_freq(h_freq2)    
        of3 = self.down2_of(of2)   
        l_freq3, h_freq3, of3 = self.cs3((l_freq3, h_freq3, of3))
        

        l_freq4 = self.down3_l_freq(l_freq3)       
        h_freq4 = self.down3_h_freq(h_freq3)    
        of4 = self.down3_of(of3)   
        l_freq4, h_freq4, of4 = self.cs4((l_freq4, h_freq4, of4))
        
        #Fuse features from all branches
        fused_features = torch.cat((l_freq4, h_freq4, of4), dim=1)
        fused_features = self.fusion_conv(fused_features)
        fused_features = self.se(fused_features)


        # ========== DECODER ==========
        # Use skip connections from low-freq + high-freq branches
        x = self.up1(fused_features, l_freq3+h_freq3) 
        x = self.up2(x, l_freq2+h_freq2)              
        x = self.up3(x, l_freq1+h_freq1)               
        reconstructed = self.outc(x)
        
        # Apply activation
        reconstructed = torch.tanh(reconstructed)
        
          # ========== PREPARE FEATURES FOR LOSS COMPUTATION ==========
        # Normalize features for contrastive/similarity losses
        features = {
            'l_freq4': F.normalize(l_freq4, p=2, dim=1),
            'h_freq4': F.normalize(h_freq4, p=2, dim=1),
            'of4': F.normalize(of4, p=2, dim=1),
        }
        
        return reconstructed, features


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Used : ",device)
    
    #khedmna hed x1, x2, x3 bech nchofouh l model y9der yinstancia w ykhdem
    x1 = torch.ones([4, 12, 256, 256]).cuda()
    x2 = torch.ones([4, 12, 256, 256]).cuda()
    x3 = torch.ones([4, 6, 256, 256]).cuda()
    model = UNet(12, 12, 6, 3).cuda()
