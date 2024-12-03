import pandas as pd
import numpy as np

# Đọc tập tin CSV
df = pd.read_csv('CIC-IDS-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv')

# Tiền xử lý dữ liệu
from sklearn.preprocessing import LabelEncoder

# Chuyển các nhãn dạng văn bản sang dạng số
encoder = LabelEncoder()
df[' Label'] = encoder.fit_transform(df[' Label'])

# Kiểm tra Dataframe không chứa giá trị null hoặc vô cùng
df = df.fillna(0)  # Replace NaN with 0
df = df.replace([np.inf, -np.inf], 0)
# print(df.isnull().sum())

df = df.astype(int)

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

X = df.drop(' Label', axis = 1)
y = df[' Label']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Thay thế dữ liệu null bằng giá trị trung bình
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

num_columns = df.shape[1]

# Trích xuất ra k cột tốt nhất để phân loại
k = min(10, num_columns)
k_best = SelectKBest(score_func = f_classif, k = k)

X_new = k_best.fit_transform(X_imputed, y)

# Trả về mảng bool, 1 tương ứng các cột được chọn và 0 nếu ngược lại
selected_features_mask = k_best.get_support()
# Trả về mảng chứa cột được chọn
elected_feature_names = X.columns[selected_features_mask]

# print(elected_feature_names)

new_columns = [' Destination Port', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packets/s', ' Min Packet Length',
       ' PSH Flag Count', ' URG Flag Count', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' min_seg_size_forward']

# New Dataframe
df_new = X[new_columns]

# Thêm nhãn vào new dataframe
df_new['label'] = df[' Label']

X1 = df_new.iloc[:, :-1].values
y1 = df_new.iloc[:, -1].values

from sklearn.model_selection import train_test_split
import tensorflow as tf

# Chia tập dữ liệu thành 2 phần train (70%) và test (30%)
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Xây dựng mô hình mạng Neuron
ann = tf.keras.models.Sequential()   # Tạo mô hình tuần tự
ann.add(tf.keras.layers.Dense(units=10,activation='sigmoid'))
ann.add(tf.keras.layers.Dense(units=10,activation='sigmoid'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer ='adam' , loss='categorical_crossentropy' , metrics= ['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

ann.fit(X_train,y_train,batch_size=32,epochs=10,callbacks=[early_stopping])

# print(ann.predict(np.array([[3268, 72, 72, 0, 0, 0, 0, 201, 72, 32]])))

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size=0.3, random_state=42)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error:",  mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

