import torch #Khối Huấn luyện
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from models.fjepa_core import FJEPA_Model
from utils.fft_transform import image_to_fft, apply_frequency_mask
from utils.dataset import FJEPA_Dataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FJEPA_Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4) 
    scaler = GradScaler()
    
    dataset_path = "dataset/DIV2K_train_HR"
    dataset = FJEPA_Dataset(image_dir=dataset_path, image_size=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    epochs = 50 # THIẾT LẬP CHẠY 50 VÒNG
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            optimizer.zero_grad()
            
            fft_spectrum = image_to_fft(real_images)
            context_freq, target_freq = apply_frequency_mask(fft_spectrum, mask_ratio=0.5)
            
            with autocast():
                s_y_pred, s_y_true, predicted_high_freq = model(context_freq, target_freq)
                latent_loss = F.mse_loss(s_y_pred, s_y_true)
                recon_loss = F.mse_loss(predicted_high_freq.real, target_freq.real) + F.mse_loss(predicted_high_freq.imag, target_freq.imag)
                loss = latent_loss + 0.5 * recon_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                    param_k.data.mul_(0.996).add_((1 - 0.996) * param_q.detach().data)
                    
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Xong! Tổng Loss: {total_loss/len(dataloader):.4f}")
        
        # CƠ CHẾ BẢO HIỂM: LƯU VÀO GOOGLE DRIVE MỖI 10 VÒNG
        if (epoch + 1) % 10 == 0:
            save_path = f"/content/drive/MyDrive/PRJET1/XLTHS/fjepa_weights_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f" ĐÃ LƯU BẢN SAO LƯU TẠI: {save_path}")

if __name__ == "__main__":
    train()
