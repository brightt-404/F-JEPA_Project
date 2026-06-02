# Xây dựng và nâng cao chất lượng ảnh dựa trên can thiệp miền tần số (F-JEPA)

Đồ án môn học Xử lý tín hiệu số. Giải pháp thay thế các mô hình Super-Resolution truyền thống bằng cách can thiệp trực tiếp vào phổ 2D-FFT kết hợp kiến trúc tự giám sát không gian ẩn (JEPA).

##  Tính năng nổi bật
* **Biến đổi Fourier 2D:** Phân tách và cô lập dải tần số thấp (Context) và dải tần số cao (Target).
* **Kiến trúc F-JEPA:** Khắc phục hiện tượng mờ ảnh (Oversmoothing) bằng cách nội suy đặc trưng miền tần số trong không gian Latent.
* **Tối ưu hóa phần cứng:** Giảm thiểu số lượng tham số học sâu, cho tốc độ suy luận nhanh trên GPU T4.

##  Cấu trúc Repository
* `/models`: Chứa mã nguồn mạng nơ-ron (Encoder, Predictor, Decoder).
* `/train_gpu.py`: Script huấn luyện mô hình tích hợp Mixed Precision.
* `BAOCAO_XLTHS.pdf`: Báo cáo chi tiết đồ án.
* `Slide_Baove.pptx`: Slide thuyết trình bảo vệ.

##  Công nghệ sử dụng
* Python 3, PyTorch
* 2D-FFT (Complex-Tensor)
* Dataset: DIV2K (được tiền xử lý cắt phổ 50%)
