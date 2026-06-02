import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.fjepa_core import FJEPA_Model
from utils.fft_transform import image_to_fft, apply_frequency_mask, fft_to_image

def super_resolution_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo mô hình (Trong thực tế sẽ load file weights .pth sau khi train DIV2K)
    model = FJEPA_Model().to(device)
    model.eval() 

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    import os
    test_img_path = os.path.join("dataset/train", os.listdir("dataset/train")[0])
    img_tensor = transform(Image.open(test_img_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Chuyển sang miền Tần số
        fft_spectrum = image_to_fft(img_tensor)
        
        # 2. Tạo ảnh mờ (Mất tần số cao)
        context_freq, _ = apply_frequency_mask(fft_spectrum, mask_ratio=0.5)
        blurry_img = fft_to_image(context_freq)
        
        # 3. AI nhúng tay vào dự đoán tần số cao
        _, _, predicted_high_freq = model(context_freq)
        
        # 4. Gắn tần số cao AI dự đoán vào vùng bị mất (Rìa ngoài)
        # Lấy mặt nạ ngược (vùng rìa = 1, vùng trung tâm = 0)
        B, C, H, W = fft_spectrum.shape
        mask = torch.ones_like(fft_spectrum, dtype=torch.float32)
        keep_h, keep_w = int(H * 0.5), int(W * 0.5)
        center_h, center_w = H // 2, W // 2
        mask[:, :, center_h - keep_h//2 : center_h + keep_h//2, 
                   center_w - keep_w//2 : center_w + keep_w//2] = 0.0
        
        # Ghép phổ: Trung tâm (Context gốc) + Rìa ngoài (AI dự đoán x Mask)
        reconstructed_spectrum = context_freq + (predicted_high_freq * mask)
        
        # 5. Nghịch đảo IFFT để ra ảnh Siêu nét
        super_res_img = fft_to_image(reconstructed_spectrum)

    # Hiển thị kết quả so sánh
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(blurry_img[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[0].set_title("1. Ảnh Đầu vào (Mờ / Low-Res)")
    axes[0].axis('off')

    axes[1].imshow(super_res_img[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[1].set_title("2. F-JEPA Khôi phục (Super-Res)")
    axes[1].axis('off')
    
    axes[2].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[2].set_title("3. Ảnh Gốc để so sánh (Ground Truth)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('thanh_qua_cuoi_cung.png', dpi=300)
    print("✅ Đã xuất bản thành phẩm Siêu độ phân giải: thanh_qua_cuoi_cung.png")

if __name__ == "__main__":
    super_resolution_test()
