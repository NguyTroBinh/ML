import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity   # Do similarity
from scipy import sparse  # Xu ly ma tran thua

# Utility Matrix: 
# Cot 0 - user, cot 1 - item, cot 2 - rating

class CF(object):
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        # Y_data: Dữ liệu đánh giá đầu vào
        # k: Số lượng láng riềng gần nhất cần xem xét để gợi ý
        # dist_func: Hàm đo khoảng cách giữa 2 vector

        self.uuCF = uuCF  # user- user CF (1), item - item CF (0)
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k    # neightbor
        self.dist_func = dist_func
        self.Ybar_data = None  # Ma trận chuẩn hóa
        # Number of users and items
        # Lấy ra chỉ số cao nhất giữa các cột (user và item)
        # Chuyển từ iterater sang int và cộng thêm 1
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    # Khi có dữ liệu mới, cập nhật tility Matrix bằng cách thêm hàng này vào cuối cùng của ma trận
    # Giả sử, không có user hay items mới, cũng không có rating nào thay đổi
    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)

    # Normalized utility matrix
    def normalize_Y(self):
        users = self.Y_data[:, 0]   # Lấy ra tất cả người dùng đã đánh giá
        self.Ybar_data = self.Y_data.copy()   # Ma trận chuẩn hóa
        self.mu = np.zeros((self.n_users, ))  # Lưu giá trị trung bình của các điểm đánh giá cho người dùng
        for n in range (self.n_users):
            # Tìm tất cả chỉ số mà tại mục user n đã đánh giá
            ids = np.where(users == n)[0].astype(np.int32)
            # Lấy ra tất cả item mà user n đã đánh giá
            item_ids = self.Y_data[ids, 1]
            # Lấy ra các rating tương ứng
            ratings = self.Y_data[ids, 2]
            # Tính trung bình của rating
            m = np.mean(ratings)
            # Nếu m là nan (tức tại đó tương ứng dấu ?) thì đặt m = 0
            if np.isnan(m):
                m = 0
            self.mu[n] = m

            # Chuẩn hóa các điểm đánh giá
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

            # Do số lượng user đã đánh giá items là rất ít, ma trận có thể chứa rất nhiều rating = 0
            # Với số lượng user và item lớn, việc lưu trữ và tính toán là không thể
            # Cần chuẩn hóa ma trận, chỉ lưu các giá trị khác 0

            self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2], (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
            # self.Ybar_data[:, 2]: Mảng chứa các giá trị được sử dụng làm phần tử khác 0 trong ma trận thưa
            # (self.Ybar_data[:, 1], self.Ybar_data[:, 0]): Xác định vị trí tương ứng với từng phần tử trong ma trận thưa
            # shape=(self.n_items, self.n_users): Chỉ định kích thước của ma trận thưa
            self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    # Cập nhật lại dữ liệu khi có dữ liệu mới
    def refresh(self):
        #
        self.normalize_Y()
        self.similarity()

    def fit(self):
        self.refresh()

    def __pred(self, u, i, normalized = 1):
        # Dự đoán rating của user u cho item i
        # Lấy ra các items i đã được đánh giá
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Lấy ra người dùng đã đánh giá items i đó
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Trả về độ tương đồng giữa các user đã đánh giá i với user u
        sim = self.S[u, users_rated_i]
        # Tìm k user có độ tương đồng gần nhất so với u
        a = np.argsort(sim)[-self.k:]
        # Chứa độ tương đồng ứng với k người dùng gần nhất
        nearest_s = sim[a]
        # Lấy ra các rating mà các user này đã đánh giá i
        r = self.Ybar[i, users_rated_i[a]]

        # Tính rating của u cho i
        if normalized:
            return (r * nearest_s)[0] /(np.abs(nearest_s).sum() + 1e-8)
            # Cộng thêm 1 hằng số nhỏ để tránh chia cho 0
        return (r * nearest_s)[0] /(np.abs(nearest_s).sum() + 1e-8) + self.mu[u] 
        # Nếu chưa chuẩn hóa: cộng thêm giá trị trung bình tương ứng với user để về ma trận gốc

    def pred(self, u, i, normalized = 1):
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)

    def recommend(self, u):
        # Tìm tất cả các items nên được gợi ý cho user u
        # Lấy ra tất cả các chỉ số của user u trong ma trận Y_data
        ids = np.where(self.Y_data[:, 0] == u)[0]
        # Lấy ra các items mà user u này đã đánh giá
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommend_items = []
        for i in range(self.n_items):
            # Nếu item i chưa được user u đánh giá, ta sẽ dự đoán rating của u cho i
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0:
                    # Nếu rating lớn hơn 0, ta sẽ gợi ý item này cho u
                    recommend_items.append(i)
        return recommend_items 

    def print_recommend(self):
        # In ra tất cả các item đã được gợi ý cho từng user
        print('Recommendation: ')
        for u in range(self.n_users):
            recommend_items = self.recommend(u)
            if self.uuCF:
                print('Recommend item(s): ', recommend_items, 'for user ', u)
            else:
                print('Recommend item', u, 'for user(s): ', recommend_items)


# Áp dụng lên dữ liệu nhỏ trước khi áp dụng cho csdl MovieLens 100K
# # Data file
# r_cols = ['user_id', 'item_id', 'rating']
# ratings = pd.read_csv('Neighborhood-Based Collaborative Filtering/ex.dat', sep = ' ', names = r_cols, encoding = 'latin-1')
# Y_data = ratings.values

# # Áp dụng với user-user CF
# rs = CF(Y_data, k = 2, uuCF = 1)
# rs.fit()

# rs.print_recommend()

# # Áp dụng với item-item CF
# rs = CF(Y_data, k = 2, uuCF = 0)
# rs.fit()

# rs.print_recommend()

# Áp dụng lên MovieLens 100K
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('Content - based Recomendation Systems/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('Content - based Recomendation Systems/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

# Trừ chỉ số của User (cột 0) và Item (cột 1) đi 1
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

from math import sqrt

# Kết quả với user - user CF:
rs = CF(rate_train, k = 30, uuCF = 1)
rs.fit()

n_tests = rate_test.shape[0]
SE = 0
for n in range(n_tests):
    # Rating được dự đoán
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2]) ** 2
RMSE = np.sqrt(SE / n_tests)
print('User-user CF, RMSE = ', RMSE)

# Kết quả với Item_item CF
rs = CF(rate_train, k = 30, uuCF = 0)
rs.fit()

n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 

RMSE = np.sqrt(SE/n_tests)
print ('Item-item CF, RMSE =', RMSE)

# Nhận xét: Item - item CF có lỗi nhỏ hơn User - user CF và tốt hơn so với Content-based Recommendation Systems