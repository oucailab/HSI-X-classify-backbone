from torch import nn
import torch
import torch.nn.functional as F
import math

def generate_cosine_positional_encoding(d_model, height, width):
    pe = torch.zeros(d_model, height, width)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_y, pos_x = torch.meshgrid(torch.arange(0., height), torch.arange(0., width))
    pe[0::2, :, :] = torch.sin(pos_y * div_term[:, None, None])
    pe[1::2, :, :] = torch.cos(pos_x * div_term[:, None, None])
    return pe


class CAFF(nn.Module):
    def __init__(self, in_channels, num_stages):
        super(CAFF, self).__init__()
        self.num_stages = num_stages
        self.attention_stages_k1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        
        self.attention_stages_k3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        """
        self.attention_stages_k3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        """
        self.attention_stages_k7 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        
        self.compression_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels*3, in_channels, kernel_size=1),
            )
            for _ in range(num_stages)
        ])
        
        
        self.proj = nn.Conv2d(in_channels, 64, kernel_size=1)
        

    def forward(self, x1, x2):  
        # hack : future implementation with learned pos_encoding ? 
        pos_encoding1 = generate_cosine_positional_encoding(x1.size(1), x1.size(2), x1.size(3))
        pos_encoding2 = generate_cosine_positional_encoding(x2.size(1), x2.size(2), x2.size(3))
        
        # additive (hack ?) fused pos_encoding 
        combined_pos_encoding = pos_encoding1 + pos_encoding2
        combined_pos_encoding = combined_pos_encoding.to(x1.device)
        
        
        for i in range(self.num_stages): # loop for hierarchical self/cross attention steps 
        # hierarchical attention + learned attention fusion (convolution based procedure)
            query_conv1, key_conv1, value_conv1, softmax1 = self.attention_stages_k1[i]
            query1 = query_conv1(x1) + combined_pos_encoding
            key1 = key_conv1(x2) + combined_pos_encoding
            value1 = value_conv1(x2)
            attn_weights1 = softmax1(torch.matmul(query1, key1.transpose(-1, -2)))
            fused_features_k1 = torch.matmul(attn_weights1, value1)
            
            query_conv3, key_conv3, value_conv3, softmax3 = self.attention_stages_k3[i]
            query3 = query_conv3(x1) 
            key3 = key_conv3(x2)
            value3 = value_conv3(x2)
            attn_weights3 = softmax3(torch.matmul(query3, key3.transpose(-1, -2)))
            fused_features_k3 = torch.matmul(attn_weights3, value3)
            
            query_conv7, key_conv7, value_conv7, softmax7 = self.attention_stages_k7[i]
            query7 = query_conv7(x1) 
            key7 = key_conv7(x2) 
            value7 = value_conv7(x2)
            attn_weights7 = softmax7(torch.matmul(query7, key7.transpose(-1, -2)))
            fused_features_k7 = torch.matmul(attn_weights7, value7)

            #  target features shape 
            target_height = fused_features_k1.size(2)
            target_width = fused_features_k1.size(3)

            # reshaping the different attention maps before concatenation : interpolation (not-learned) 
            k3_resized = F.interpolate(fused_features_k3, size=(target_height, target_width), mode='bilinear', align_corners=False)
            k7_resized = F.interpolate(fused_features_k7, size=(target_height, target_width), mode='bilinear', align_corners=False)

            fused_features = torch.cat((fused_features_k1, k3_resized, k7_resized), 1) # stacking 
            fused_features = self.compression_convs[i](fused_features) # learned operation: compress to the standard features shape 
            
            # Update x1 and x2 for the next stage
            x1, x2 = fused_features, fused_features
            
        # projects to the reduced dimensionnality (legacy)
        #x1_conv = self.vis_proj(x1_value)
        #x2_conv = self.ir_proj(x2_value)
        
        # features concatenation (legacy) 
        #features = torch.cat((x1_conv, x2_conv, fused_proj), 1)
        
        fused_features = self.proj(fused_features) # projection : learned reshaping for dimensionality compression
        return x1, x2