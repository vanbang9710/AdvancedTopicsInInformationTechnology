# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import NcFactory

# Đọc dữ liệu từ file ys1a.csv
dfbi = pd.read_csv('ys1a.csv')

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Lấy các biến độc lập (vec, delta, deltachi, deltahmix, deltasmix) và biến phụ thuộc (ys) từ dữ liệu
X_train = train_data[['vec', 'delta', 'deltachi', 'deltahmix', 'deltasmix']].values
y_train = train_data['ys'].values
X_test = test_data[['vec', 'delta', 'deltachi', 'deltahmix', 'deltasmix']].values
y_test = test_data['ys'].values

# Khởi tạo một mô hình hồi quy tuyến tính
model = LinearRegression()

# Đóng gói mô hình hồi quy tuyến tính thành một RegressorAdapter để sử dụng với nonconformist
model = RegressorAdapter(model)

# Tạo một NcFactory để sử dụng phương pháp hậu xử lý Least Squares Prediction Interval (LSPI)
# để tính toán mức độ không phù hợp (nonconformity score) cho mỗi điểm dữ liệu
nc = NcFactory.create_nc(model, err_func=mean_squared_error)

# Tạo một IcpRegressor để thực hiện conformal prediction với NcFactory đã tạo
icp = IcpRegressor(nc)

# Huấn luyện IcpRegressor với tập huấn luyện
icp.fit(X_train, y_train)

# Dự đoán khoảng dự báo cho tập kiểm tra với mức tin cậy 0.95
prediction = icp.predict(X_test, significance=0.05)

# In ra kết quả dự đoán
print('Kết quả dự đoán:')
print('Giá trị thực | Khoảng dự báo')
for i in range(len(y_test)):
    print(f'{y_test[i]:.2f} | ({prediction[i, 0]:.2f}, {prediction[i, 1]:.2f})')
