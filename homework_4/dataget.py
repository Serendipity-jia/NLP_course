# from sklearn.datasets import fetch_20newsgroups

# # 本地下载并缓存数据集
# data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
# print("本地数据集下载完成！")
from sklearn.datasets import get_data_home
print("数据集缓存路径：", get_data_home())
from sklearn.datasets import fetch_20newsgroups

# 直接读取本地缓存，不联网
data = fetch_20newsgroups(
    subset='all', 
    remove=('headers', 'footers', 'quotes'),
    download_if_missing=False  # 关键：不尝试联网下载
)