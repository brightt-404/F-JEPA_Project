import torch
import torch.nn.functional as F
from models.fjepa_core import FJEPA_Model
from utils.fft_transform import image_to_fft, apply_frequency_mask

def debug_on_cpu():
    print("⏳ Đang khởi tạo mô hình trên CPU...")
    device = torch.device("cpu") 
    
    try:
        model = FJEPA_Model().to(device)
        optimizer = torch.optim.AdamW(model.context_encoder.parameters(), lr=1e-4)
        
        print("⏳ Đang tạo tensor dữ liệu giả (64x64) để chống tràn RAM...")
        dummy_images = torch.rand(1, 3, 64, 64).to(device)
        
        optimizer.zero_grad()
       
        print("⏳ Đang tính toán 2D-FFT và tạo Mask...")
        fft_spectrum = image_to_fft(dummy_images)
        context_freq, target_freq = apply_frequency_mask(fft_spectrum, mask_ratio=0.5)
        
        print("⏳ Đang chạy luồng dự đoán (Forward)...")
        s_y_pred, s_y_true = model(context_freq, target_freq)
        
        loss = F.mse_loss(s_y_pred, s_y_true)
        
        print("⏳ Đang tính toán đạo hàm (Backward)...")
        loss.backward()
        optimizer.step()
        
        print("-" * 50)
        print(f"✅ THÀNH CÔNG! Module DSP và mạng Nơ-ron đã liên kết khớp nhau.")
        print(f"✅ Loss khởi tạo: {loss.item():.4f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Lỗi rồi, kiểm tra lại: {e}")

if __name__ == "__main__":
    debug_on_cpu()