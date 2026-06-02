import torch # Khối Mạng Nơ-ron (AI)
import torch.nn as nn
import copy

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=6, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # Nối phần thực và ảo
        x_combined = torch.cat([x.real, x.imag], dim=1)
        return self.net(x_combined)

class SimpleDecoder(nn.Module):
    def __init__(self, in_dim=256, out_channels=6):
        super().__init__()
        # Giải mã từ Không gian ẩn (Latent) về lại Phổ tần số
        # Sử dụng ConvTranspose2d (Tích chập chuyển vị) để phóng to ma trận
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, latent):
        out = self.net(latent)
        # Cắt làm đôi: 3 kênh đầu là phần Thực (Real), 3 kênh sau là phần Ảo (Imag)
        real_part, imag_part = torch.chunk(out, 2, dim=1)
        # Ghép lại thành Số Phức chuẩn DSP
        return torch.complex(real_part, imag_part)

class FJEPA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = SimpleEncoder()
        
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        
        # MẢNH GHÉP CUỐI CÙNG: DECODER
        self.decoder = SimpleDecoder()

    def forward(self, context_freq, target_freq=None):
        s_x = self.context_encoder(context_freq)
        s_y_pred = self.predictor(s_x)
        
        # Decoder dịch Latent thành dải phổ cao ảo (Predicted High Freq)
        predicted_high_freq = self.decoder(s_y_pred)
        
        if target_freq is not None:
            with torch.no_grad():
                s_y_true = self.target_encoder(target_freq)
            return s_y_pred, s_y_true, predicted_high_freq
            
        return s_y_pred, None, predicted_high_freq
