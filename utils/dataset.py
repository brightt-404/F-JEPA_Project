import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FJEPA_Dataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        # Tìm tất cả file ảnh trong thư mục
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Tiền xử lý: Ép về kích thước chuẩn và biến thành Tensor
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image
