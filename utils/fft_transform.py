import torch  # Khối xử lý tín hiệu
import torch.fft

def image_to_fft(image_tensor):
    """Biến đổi thuận: Không gian -> Tần số"""
    fft_spectrum = torch.fft.fft2(image_tensor, dim=(-2, -1), norm="ortho")
    fft_shifted = torch.fft.fftshift(fft_spectrum, dim=(-2, -1))
    return fft_shifted

def fft_to_image(fft_shifted):
    """Biến đổi nghịch: Tần số -> Không gian (IFFT)"""
    fft_unshifted = torch.fft.ifftshift(fft_shifted, dim=(-2, -1))
    image_tensor = torch.fft.ifft2(fft_unshifted, dim=(-2, -1), norm="ortho")
    # Lấy giá trị tuyệt đối (biên độ thực) để hiển thị thành điểm ảnh
    return torch.abs(image_tensor)

def apply_frequency_mask(fft_spectrum, mask_ratio=0.5):
    """Tạo mặt nạ tách dải tần thấp (Context) và cao (Target)"""
    B, C, H, W = fft_spectrum.shape
    center_h, center_w = H // 2, W // 2
    
    keep_h, keep_w = int(H * mask_ratio), int(W * mask_ratio)
    mask = torch.zeros_like(fft_spectrum, dtype=torch.float32)
    
    h_start, h_end = center_h - keep_h // 2, center_h + keep_h // 2
    w_start, w_end = center_w - keep_w // 2, center_w + keep_w // 2
    
    mask[:, :, h_start:h_end, w_start:w_end] = 1.0
    
    context_spectrum = fft_spectrum * mask
    target_spectrum = fft_spectrum * (1 - mask)
    
    return context_spectrum, target_spectrum
