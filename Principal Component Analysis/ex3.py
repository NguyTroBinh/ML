# Eigenface

import numpy as np
import imageio    # for loading image

path = 'Principal Component Analysis/unpadded/'
ids = range(1, 16)  # 15 person
# Tất cả các trạng thái của 1 person
states = ['centerlight', 'glasses', 'happy', 'leftlight', 
          'noglasses', 'normal', 'rightlight','sad', 
          'sleepy', 'surprised', 'wink' ]
prefix = 'subject'
surfix = '.pgm'

# Chiều cao và chiều rộng bức ảnh
h = 116
w = 98
D = h * w  # Số lượng pixel trong 1 bức ảnh
N = len(states) * 15

# Khởi tạo mảng chứa dữ liệu ảnh, mỗi cột của X sẽ chứa 1 hình ảnh được làm phẳng
X = np.zeros((D, N))

cnt = 0
for person_id in range (1, 16):
    for state in states:
        # Đường dẫn tới bức ảnh
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        # zfill(k): Lấy ra k kí tự đầu tiên, ở đây ta muốn lấy ra id của person
        # Đọc ảnh và làm phẳng về 1 chiều
        X[:, cnt] = imageio.v2.imread(fn).reshape(D)
        cnt += 1

# Sử dụng PCA để giảm chiều dữ liệu
from sklearn.decomposition import PCA

pca = PCA(n_components = 100)  # Ở đây ta chọn K = 100, tức số thành phần muốn giữ lại
# Cần chuyển dữ liệu theo hàng trước khi fit, do sklearn chỉ làm việc với dữ liệu theo hàng
pca.fit(X.T)

U = pca.components_.T   # Ma trận thu được sau khi giảm số chiều, có kích thước (D, K)

import matplotlib.pyplot as plt

for i in range(U.shape[1]):
    plt.axis('off')
    # Hiển thị từng eigenface. Chuyển từ vector có số chiều 116 * 98 sang ma trận 2 chiều (116, 98) tức ảnh gốc
    # interpolation='nearest': Không thực hiện nội suy. Tức giữ nguyên hình ảnh, không thay đổi gì
    f1 = plt.imshow(U[:, i].reshape(116, 98), interpolation='nearest')
    # Ẩn các trục x và y 
    f1.axes.get_xaxis().set_visible(False)
    f1.axes.get_yaxis().set_visible(False)
    plt.gray()   # Hiển thị dưới dạng thang độ xám
    fn = 'eigenface' + str(i).zfill(2) + '.png'
    plt.savefig(fn, bbox_inches = 'tight', pad_inches = 0)
    # bbox_inches = 'tight': Cắt các khoảng trống thừa xung quanh hình ảnh
    # pad_inches = 0: Đảm bảo không có khoảng trắng thêm vào

for person_id in range(1, 7):
    for state in ['centerlight']:
        # Đọc và hiển thị ảnh gốc
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        im = imageio.v2.imread(fn)
        plt.axis('off')
        f1 = plt.imshow(im, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        fn = 'ori' + str(person_id).zfill(2) + '.png'
        plt.savefig(fn, bbox_inches = 'tight', pad_inches = 0)
        plt.show()

        # Chuẩn hóa ảnh bằng cách trừ đi giá trị trnng bình để đưa dữ liệu về dạng có trung bình bằng 0
        x = im.reshape(D, 1) - pca.mean_.reshape(D, 1)

        z = U.T.dot(x)
        # Chuyển đổi ảnh từ không gian PCA về không gian gốc
        x_tilde = U.dot(z) + pca.mean_.reshape(D, 1)
        # Hiển thị ảnh tái tạo
        im_tilde = x_tilde.reshape(116, 98)
        plt.axis('off')
        f1 = plt.imshow(im_tilde, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        fn = 'res' + str(person_id).zfill(2) + '.png'
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.show()