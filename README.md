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
8. **Deepface** (0.0.81) <br>
Thư viện Deepface dùng để xác định vị trí khuôn mặt <br>
*Lệnh cài đặt* <br>
```
pip install deepface
```
9. **Colormath** (3.0.0) (optional) <br>
Thư viện Colormath dùng để xác định skintone mà không cần qua huấn luyện mô hình <br>
*Lệnh cài đặt* <br>
```
pip install colormath
```
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
