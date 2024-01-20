# AI Hackathon: Face Analysis Challenge 
# (NTN Team)
## Team members:
Trần Quang Nhật [github](https://github.com/Yamakaze-chan)   
Võ Thành Thái [github](https://github.com/thaivo02)   
Nguyễn Duy Nhật [github](#)   
<br>

# Introduction
Đây là bài dự thi cuộc thi AI Hackathon: Face Analysis Challenge.<br>
Ứng dụng các công nghệ Trí tuệ Nhân tạo để xác định tất cả các đặc điểm của khuôn mặt con người bao gồm: 
* Giới tính
* Độ tuổi
* Dân tộc
* Màu da
* Đeo khẩu trang/ Không đeo khẩu trang
* Cảm xúc
* Vị trí khuôn mặt

# Thư viện sử dụng
1. **OpenCV** (4.6.0.66)<br>
Thư viện OpenCV chủ yếu được sử dụng để đọc và xử lý ảnh <br>
*Lệnh cài đặt* <br>
```
pip install opencv-python
```
2. **Tensorflow** (2.14.0) <br>
Thư viện Tensorflow (cụ thể là Keras) dùng để xây dựng các mạng neural <br>
*Lệnh cài đặt* <br>
```
pip install tensorflow
```
3. **Pandas** (2.0.2) <br>
Thư viện Pandas dùng để đọc dữ liệu từ file .csv <br>
*Lệnh cài đặt* <br>
```
pip install pandas
```
4. **Numpy** (1.26.0) <br>
Thư viện Numpy dùng để xử lý các dạng số học (ví dụ như ma trận) <br>
*Lệnh cài đặt* <br>
```
pip install numpy
```
5. **Imbalanced-learn** (0.11.0) <br>
Thư viện Imbalanced-learn dùng để cân bằng lại dữ liêu <br>
*Lệnh cài đặt* <br>
```
pip install imblearn
```
6. **Scikit-learn** (1.2.2) <br>
Thư viện Scikit-learn dùng để chia lại dữ liêu (train, test) <br>
*Lệnh cài đặt* <br>
```
pip install scikit-learn
```
7. **Matplotlib** (3.8.2) <br>
Thư viện Matplotlib dùng để lưu lại kết quả huấn luyện mô hình dưới dạng hình ảnh <br>
*Lệnh cài đặt* <br>
```
pip install matplotlib
```
8. **Colormath** (3.0.0) (optional) <br>
Thư viện Colormath dùng để phân loại skintone mà không cần qua huấn luyện mô hình <br>
*Lệnh cài đặt* <br>
```
pip install colormath
```
# Các file cần thiết
[Pretrained Yunet](https://github.com/ShiqiYu/libfacedetection.train/tree/master/onnx) <br>
[Pretrained facial expression](https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5)
<br>
# Hướng dẫn sử dụng
**Bước 1:** <Br>
Tải về dưới dạng file .zip hoặc sử dụng các câu lệnh: <br>
```
git clone https://github.com/thaivo02/Facial-recognition.git
cd Facial-recognition
```
**Bước 2** <br>
Chạy với 1 đường dẫn ảnh bất kì <br>
```
python main.py
```
Chạy realtime với webcam:
```
python webcam_detect.py
```
# Cấu trúc thiết kế 
## Thư mục
```
project_base_path
└───  other_files
            | yunet_n_640_640.onnx <mô hình Yunet>
            | facial_expression_model_weights.h5
└───  <Đặc điểm khuôn mặt>
        └───  models
                └───  <các model đã huấn luyện>
        └───  plot
                |   <Hình ảnh lưu kết quả huấn luyện>
        | <đặc điểm khuôn mặt>_predict.py
        | <đặc điểm khuôn mặt>_train.py
        ...
└───  balance_data
            |   down_sampling.py
extract_frontface.py
extract_skin.py
main.py
webcam_detect.py
```
## File huấn luyện mô hình *(<đặc điểm khuôn mặt>_train.py)*<br> 
train\_<đặc điểm khuôn mặt>_model(nrow=0): Hàm chính, dùng để đọc dữ liệu, xử lý dữ liệu và huấn luyện mô hình <br>
train\_<đặc điểm khuôn mặt>_model(): Hàm dùng để thiết kế mạng (CNN hoặc các mạng tương tự) <br>
load\_images\_from_folder(list_data): Xử lý hình ảnh để chuẩn bị huấn luyện mô hình
* nrow : Số lượng dòng đọc từ file csv (sử dụng bao nhiêu ảnh để train), 0 tương ứng là đọc hết
* list_data: dữ liệu trong file csv (gồm tên file và label)
<br>
## File dự đoán *(<đặc điểm khuôn mặt>_predict.py)*<br> 
predict\_<đặc điểm khuôn mặt>(img, model): hàm dùng để trả về kết quả dự đoán của mô hình<br>
* img: Ảnh đầu vào
* model: đường dẫn đến mô hình đã huấn luyện
<br>
### * Lưu ý * <br>
Đối với skintone\_predict thì hàm chỉ nhận 1 ảnh làm input (skintone_predict(image))<br>
# Kết quả huấn luyện mô hình<br>
## Giới tính<br>
![Gender Accuracy](https://github.com/thaivo02/Facial-recognition/blob/main/gender/plot/gender_CNN_plot_acc.png?raw=true)<br>
## Tuổi<br>
![Age Accuracy](https://github.com/thaivo02/Facial-recognition/blob/main/age/plot/age_CNN_plot_acc1.png?raw=true)<br>
## Cảm xúc<br>
Vì sử dụng pretrained model nên không đánh giá<br>
## Chủng tộc<br>
![Race Accuracy](https://github.com/thaivo02/Facial-recognition/blob/main/ethnicity/plot/ethnicity_CNN_plot_acc.png?raw=true)<br>
## Đeo khẩu trang<br>
![Masked Accuracy](https://github.com/thaivo02/Facial-recognition/blob/main/mask/plot/mask_CNN_plot_acc.png?raw=true)<br>
