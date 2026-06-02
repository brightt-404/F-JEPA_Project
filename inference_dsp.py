import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.fjepa_core import FJEPA_Model
from utils.fft_transform import image_to_fft, apply_frequency_mask, fft_to_image

def analyze_signal_processing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        fft_spectrum = image_to_fft(img_tensor)
        context_freq, _ = apply_frequency_mask(fft_spectrum, mask_ratio=0.5)
        
        # THÊM BƯỚC MỚI: IFFT đảo mạch dải tần thấp để xem ảnh bị mờ ra sao
        blurry_img_tensor = fft_to_image(context_freq)
        
        mag_original = 20 * torch.log10(torch.abs(fft_spectrum[0, 0]) + 1e-8).cpu()
        mag_context = 20 * torch.log10(torch.abs(context_freq[0, 0]) + 1e-8).cpu()

    # Nâng cấp lên 4 khung hình
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("1. Ảnh Gốc (HR)")
    axes[0].axis('off')

    axes[1].imshow(mag_original.numpy(), cmap='magma')
    axes[1].set_title("2. Phổ 2D-FFT Toàn phần")
    
    axes[2].imshow(mag_context.numpy(), cmap='magma')
    axes[2].set_title("3. Phổ Context (Mất Cao tần)")

    # Hiển thị ảnh sau khi IFFT
    blurry_img_display = blurry_img_tensor[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    axes[3].imshow(blurry_img_display)
    axes[3].set_title("4. Ảnh IFFT (Bị mờ/Low-Res)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('ket_qua_DSP_full.png', dpi=300)
    print("✅ Đã lưu biểu đồ phân tích 4 bước thành file ket_qua_DSP_full.png")

if __name__ == "__main__":
    analyze_signal_processing()
