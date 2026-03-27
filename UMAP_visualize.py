import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import umap
import collections
#词汇表类，用来配合保存过的数据重建词汇表对象
class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        #扁平化处理输入列表
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        
        #统计每个token的频数
        counter = collections.Counter(tokens)#返回一个字典，键是token，值是出现频数
        #按频数降序排序
        self.token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)

        #构建唯一token列表(raw_text中也包含<unk>,所以这里要把所有token都进行去重)
        #但是不能用set(),因为它只能无序去重，而dict.fromkey()可以实现有序去重，只保留第一次出现的位置
        self.idx_to_token = list(dict.fromkeys(["<unk>"] + reserved_tokens +[
            token for token, freq in self.token_freqs if freq >= min_freq
        ]))

        #构建token→idx的映射字典
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
    #内置方法：简化调用
    def __len__(self):
        return len(self.idx_to_token)
    #查询方法：token→idx
    def __getitem__(self, tokens):
        # 处理单个token：存在则返回idx，否则返回<unk>的idx
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        #处理多个token（列表/元组）：递归调用，返回idx列表
        return [self.__getitem__(token) for token in tokens]
    #反向映射方法：idx→token
    def to_tokens(self, indices):
        #处理多个索引（只要有长度属性，例如列表、numpy数组、torch张量）：返回token列表
        if hasattr(indices, "__len__") and len(indices) > 1:
            return [self.idx_to_token[int(indice)] for indice in indices]
        #处理单个索引:返回单个token
        return self.idx_to_token[indices]
    #属性方法，便捷获取未知token索引:@property装饰器将方法转为属性，可直接用vocab.unk获取<unk>的索引
    @property
    def unk(self):
        return self.token_to_idx["<unk>"]
    
def reduce_embeddings_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42):
    """
    使用 UMAP 将高维词向量降到二维
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d

def plot_umap_embeddings(
    embeddings_2d,
    tokens,
    figsize=(7, 5),
    point_size=8,
    alpha=0.7,
    annotate_tokens=None,
    annotate_indices=None,
    annotate_every=None,
    title='UMAP Visualization of Word Embeddings',
    save_path=None
):
    """
    绘制 UMAP 降维后的词向量

    参数说明：
    - embeddings_2d: UMAP 输出的二维坐标，shape=[vocab_size, 2]
    - tokens: 与索引对应的 token 列表
    - annotate_tokens: 需要特别标注的 token 列表，例如 ['chip', 'chips']
    - annotate_indices: 需要特别标注的索引列表
    - annotate_every: 每隔多少个点标一个标签，例如 100 表示每 100 个词标一次
    - save_path: 图片保存路径，例如 'umap_words.png'
    """
    plt.figure(figsize=figsize)

    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]

    # 绘制全部点
    plt.scatter(x, y, s=point_size, alpha=alpha, c='steelblue')
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.grid(True, linestyle='--', alpha=0.3)

    # 需要标注的索引集合
    indices_to_annotate = set()

    if annotate_indices is not None:
        indices_to_annotate.update(
            idx for idx in annotate_indices if 0 <= idx < len(tokens)
        )

    if annotate_tokens is not None:
        token_to_idx = {tok: i for i, tok in enumerate(tokens)}
        for tok in annotate_tokens:
            if tok in token_to_idx:
                indices_to_annotate.add(token_to_idx[tok])

    if annotate_every is not None and annotate_every > 0:
        indices_to_annotate.update(range(0, len(tokens), annotate_every))

    # 加标签
    for idx in indices_to_annotate:
        plt.annotate(
            tokens[idx],
            (x[idx], y[idx]),
            fontsize=8,
            alpha=0.85
        )

    # 对高亮点重新画一遍，颜色更醒目
    if len(indices_to_annotate) > 0:
        highlight_x = [x[idx] for idx in indices_to_annotate]
        highlight_y = [y[idx] for idx in indices_to_annotate]
        plt.scatter(highlight_x, highlight_y, s=20, c='red', alpha=0.9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



if __name__ == '__main__':
    # 加载保存的词嵌入模型权重和词汇表
    with open("vocab_and_embedding.pkl", "rb") as f:
        load_obj = pickle.load(f)

    vocab = load_obj["vocab"]          # 词汇表类实例
    embeddings = load_obj["embedding"]  # 词嵌入向量

    #获取前500个词嵌入向量和对应的token
    embeddings = embeddings[:100]
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    vocab_size = embeddings.shape[0]
    tokens = vocab.to_tokens(list(range(vocab_size)))

    print(f'词向量矩阵形状: {embeddings.shape}')
    print(f'词表大小: {len(tokens)}')
    #UMAP降维
    embeddings_2d = reduce_embeddings_umap(embeddings)
    #绘图，设置需要标注token的索引或者间隔
    plot_umap_embeddings(embeddings_2d=embeddings_2d,tokens=tokens,annotate_every=True,
                         save_path="./word_embeddings_umap.png")



