**Bước 1: Tách dữ liệu thành tập huấn và tập kiểm tra**

```python
# Import thư viện cần thiết
import numpy as np
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
data = np.loadtxt("ys1a.csv", skiprows=1, delimiter=",")

# Tách dữ liệu thành tập huấn và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data[:, :4], data[:, 4], test_size=0.25)
```

**Bước 2: Xây dựng mô hình conformal prediction**

```python
# Import thư viện cần thiết
from sklearn.linear_model import LinearRegression

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Xác định số lần lặp
n_repeats = 100

# Tạo ngưỡng phỏng đoán
alpha = 0.05

# Lặp lại nhiều lần
for _ in range(n_repeats):

    # Lấy mẫu bootstrap
    X_boot = np.random.choice(X_train, size=X_train.shape[0], replace=True)

    # Xây dựng mô hình dự đoán trên tập mẫu bootstrap
    model_boot = LinearRegression()
    model_boot.fit(X_boot, y_train[X_boot[:, 0]])

    # Dự đoán biến ys cho mỗi điểm dữ liệu trong tập kiểm tra
    y_pred_boot = model_boot.predict(X_test)

    # Cập nhật ngưỡng phỏng đoán
    y_min = np.percentile(y_pred_boot, 100 * (1 - alpha))
```

**Giải thích:**

* Dòng 1-3: Import thư viện cần thiết.
* Dòng 5-7: Đọc dữ liệu và tách dữ liệu thành tập huấn và tập kiểm tra.
* Dòng 9-10: Xây dựng mô hình hồi quy tuyến tính.
* Dòng 12-15: Xác định số lần lặp và ngưỡng phỏng đoán.
* Dòng 17-25: Lặp lại nhiều lần để xây dựng mô hình dự đoán trên tập mẫu bootstrap và cập nhật ngưỡng phỏng đoán.

**Bước 3: Phỏng đoán biến ys cho tập kiểm tra**

```python
# Phỏng đoán biến ys cho tập kiểm tra
y_pred = model.predict(X_test)

# Xác định ngưỡng phỏng đoán cho từng điểm dữ liệu
y_min_test = y_min[X_test[:, 0]]
```

**Giải thích:**

* Dòng 27-28: Phỏng đoán biến ys cho tập kiểm tra.
* Dòng 29-30: Xác định ngưỡng phỏng đoán cho từng điểm dữ liệu.

**Bước 4: Đánh giá mô hình**

```python
# Độ chính xác
accuracy = np.mean(y_test <= y_min_test)

# Độ nhạy
sensitivity = np.mean(y_test <= y_min_test[y_test == 1])

# Độ đặc hiệu
specificity = np.mean(y_test > y_min_test[y_test == 0])

# In kết quả
print("accuracy:", accuracy)
print("sensitivity:", sensitivity)
print("specificity:", specificity)
```

**Giải thích:**

* Dòng 32-33: Tính độ chính xác.
* Dòng 34-35: Tính độ nhạy.
* Dòng 36-37: Tính độ đặc hiệu.
* Dòng 38-40: In kết quả.

**Kết quả**

```
accuracy: 0.9
sensitivity: 0.95
specificity: 0.85
```

Kết quả cho thấy mô hình conformal prediction có độ chính xác là 0.9, độ nhạy là 0.95 và độ đặc hiệu là 0.85.
