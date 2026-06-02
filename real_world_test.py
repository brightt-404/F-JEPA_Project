import torch #Khối Khôi phục Thực tế
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.fjepa_core import FJEPA_Model
from utils.fft_transform import image_to_fft, apply_frequency_mask, fft_to_image
import os

def run_real_world_image(image_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FJEPA_Model().to(device)
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(" Đã nạp trọng số thành công!")
    else:
        print(f" Không tìm thấy file {weight_path}. Hãy kiểm tra lại số Epoch!")
        return
        
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f" Không tìm thấy bức ảnh: {image_path}. Bạn đã up lên Colab chưa?")
        return

    print(" Đang phân tích phổ và nội suy chi tiết...")
    with torch.no_grad():
        fft_spectrum = image_to_fft(img_tensor)
        context_freq, _ = apply_frequency_mask(fft_spectrum, mask_ratio=0.5)
        _, _, predicted_high_freq = model(context_freq)
        
        B, C, H, W = fft_spectrum.shape
        mask = torch.ones_like(fft_spectrum, dtype=torch.float32)
        keep_h, keep_w = int(H * 0.5), int(W * 0.5)
        center_h, center_w = H // 2, W // 2
        mask[:, :, center_h - keep_h//2 : center_h + keep_h//2, 
                   center_w - keep_w//2 : center_w + keep_w//2] = 0.0
        
        reconstructed_spectrum = context_freq + (predicted_high_freq * mask)
        super_res_img = fft_to_image(reconstructed_spectrum)
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[0].set_title("Ảnh Đầu Vào (Mờ)")
    axes[0].axis('off')
    
    axes[1].imshow(super_res_img[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[1].set_title("F-JEPA Phục Hồi (Nét)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_thuc_te.png', dpi=300)
    print(" Xong! Đã lưu thành quả vào file test_thuc_te.png")

if __name__ == "__main__":
    # ĐIỀN TÊN ẢNH CỦA BẠN VÀO ĐÂY (nhớ up ảnh lên trước nhé)
   anh_cua_ban = "anh_phong_canh_mo.jpg"
file_nao_ai = "/content/drive/MyDrive/PRJET1/XLTHS/fjepa_weights_epoch_50.pth"
run_real_world_image(anh_cua_ban, file_nao_ai)
