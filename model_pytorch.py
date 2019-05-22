import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import torch.optim as optim

class SRCNN(nn.Module):
    '''
    A raw audio deep autoencoder for super resolution.
    '''

    def __init__(self, ):
        super(SRCNN, self).__init__()


        self.enc1 = nn.Conv1d(in_channels = 1,
                              out_channels = 128,
                              kernel_size = 65,
                              stride = 2,
                              padding = 32)
        self.enc2 = nn.Conv1d(128, 256, 33, 2, 16)
        self.enc3 = nn.Conv1d(256, 512, 17, 2, 8)
        self.enc4 = nn.Conv1d(512, 512, 9, 2, 4)
        self.bottleneck = nn.Conv1d(512, 512, 9, 2, 4)
        self.dec4 = nn.Conv1d(512, 512, 9, 1, 4)
        self.dec3 = nn.Conv1d(768, 512, 17, 1, 8)
        self.dec2 = nn.Conv1d(768, 256, 33, 1, 16)
        self.dec1 = nn.Conv1d(384, 128, 65, 1, 32)
        self.final = nn.Conv1d(192, 2, 9, 1, 4)

    def forward(self, x):
        '''
        Forward function
        '''

        e1 = F.leaky_relu(self.enc1(x), 0.2)
        e2 = F.leaky_relu(self.enc2(e1), 0.2)
        e3 = F.leaky_relu(self.enc3(e2), 0.2)
        e4 = F.leaky_relu(self.enc4(e3), 0.2)

        # Bottleneck Layer
        bn = F.leaky_relu(F.dropout(self.bottleneck(e4), 0.5), 0.25)
        
        # Upsampling Layers
        d4 = self.pixel_shuffle(F.relu(F.dropout(self.dec4(bn), 0.5)), 2)
        d4_c = torch.cat((d4, e4), dim = 1)
        d3 = self.pixel_shuffle(F.relu(F.dropout(self.dec3(d4_c), 0.5)), 2)
        d3_c = torch.cat((d3, e3), dim = 1)
        d2 = self.pixel_shuffle(F.relu(F.dropout(self.dec2(d3_c), 0.5)), 2)
        d2_c = torch.cat((d2, e2), dim = 1)
        d1 = self.pixel_shuffle(F.relu(F.dropout(self.dec1(d2_c), 0.5)), 2)
        d1_c = torch.cat((d1, e1), dim = 1)

        # Final Layer
        final = self.pixel_shuffle(self.final(d1_c), 2)

        return x + final
    
    def pixel_shuffle(self, inputs, upscaling_factor=2):
        batch_size, channels, in_length = inputs.shape
        channels //= upscaling_factor
        out_length = in_length * upscaling_factor
        
        input_view = inputs.contiguous().view(batch_size, channels, upscaling_factor, in_length)
        input_view = input_view.permute(0, 1, 3, 2).contiguous()
        output_view = input_view.view(batch_size, channels, out_length)
        return output_view