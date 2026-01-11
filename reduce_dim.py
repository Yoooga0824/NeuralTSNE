import numpy as np

def adjust_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def reduce_dim_method(reduce_dim_method,data,labels,n_components):
    if reduce_dim_method == 'LDA':
        import numpy as np
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        reduce_data = lda.fit_transform(data, labels).astype(np.float32)
        return reduce_data
    elif reduce_dim_method == 'T-SNE':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components, perplexity=30, random_state=42)
        reduce_data = tsne.fit_transform(data)
        return reduce_data
    elif reduce_dim_method == 'U-MAP':
        from umap import UMAP 
        # umap = UMAP(n_components=n_components, n_neighbors=1000, min_dist=0.01, metric='euclidean', random_state=42)
         # 速度优化参数
        umap = UMAP(
            n_components=n_components,
            n_neighbors=15,           # 减小近邻数（默认15）
            n_epochs=200,             # 减少迭代次数（默认500）
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
            n_jobs=-1,                # 使用并行加速
            low_memory=True,          # 低内存模式
            verbose=True              # 查看进度
        )
        reduce_data = umap.fit_transform(data) 
        return reduce_data
    elif reduce_dim_method == 'ISOMAP':    
        from sklearn.manifold import Isomap
        isomap = Isomap(
            n_components=n_components,
            n_neighbors=10,
            # n_jobs=-1,                # 使用并行加速
            neighbors_algorithm='kd_tree',  # 使用KD树加速
            n_jobs=-1                     # 单线程避免内存问题

        )
        reduce_data = isomap.fit_transform(data) 
        return reduce_data
    elif reduce_dim_method == 'LLE':
        from sklearn.manifold import LocallyLinearEmbedding
        import numpy as np
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=10, random_state=42)
        reduce_data = lle.fit_transform(data).astype(np.float32)
        return reduce_data
    elif reduce_dim_method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components)
        reduce_data = pca.fit_transform(data)
        return reduce_data
    elif reduce_dim_method == 'IncrementalPCA':
        import numpy as np
        from sklearn.decomposition import IncrementalPCA
        ipca = IncrementalPCA(n_components, batch_size=100)
        reduce_data = ipca.fit_transform(data).astype(np.float32)
        return reduce_data
    elif reduce_dim_method == 'FactorAnalysis':
        import numpy as np
        from sklearn.decomposition import FactorAnalysis
        fa = FactorAnalysis(n_components, random_state=42)
        reduce_data = fa.fit_transform(data).astype(np.float32)
        return reduce_data
    elif reduce_dim_method == 'FastICA':
        from sklearn.decomposition import FastICA
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import VarianceThreshold
        import numpy as np
        selector = VarianceThreshold(threshold=0.0)  # 移除所有样本取值相同的特征
        data = selector.fit_transform(data)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        ica = FastICA(n_components, random_state=42)
        reduce_data = ica.fit_transform(data)
        return reduce_data
    elif reduce_dim_method == 'SpectralEmbedding':
        from sklearn.manifold import SpectralEmbedding
        import numpy as np
        se = SpectralEmbedding(n_components, n_neighbors= 20, gamma=100.0)
        reduce_data = se.fit_transform(data).astype(np.float32)
        return reduce_data
    elif reduce_dim_method == 'MDS':
        from sklearn.manifold import MDS
        mds = MDS(n_components, random_state=42)
        reduce_data = mds.fit_transform(data).astype(np.float32)
        return reduce_data
    elif reduce_dim_method == 'SVD':
        from sklearn.decomposition import TruncatedSVD
        import numpy as np
        svd = TruncatedSVD(n_components)
        reduce_data = svd.fit_transform(data).astype(np.float32)
        return reduce_data
    # Adding new dimensionality reduction methods
    elif reduce_dim_method == 'LocalMap':
        from pacmap import LocalMAP
        embedder = LocalMAP(n_components=n_components, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
        reduce_data = embedder.fit_transform(data)
        return reduce_data
    elif reduce_dim_method == 'NeuralTSNE':
        import os
        import sys
        import numpy as np
        import torch
        import pytorch_lightning as L
        from torch.utils.data import DataLoader, TensorDataset

        # 支持两种使用方式：1) 已 pip 安装 NeuralTSNE；2) 直接在该仓库根目录运行脚本
        try:
            from NeuralTSNE.TSNE.ParametricTSNE import ParametricTSNE
            from NeuralTSNE.TSNE.Modules import DimensionalityReduction
        except ImportError:
            repo_root = os.path.dirname(__file__)
            local_pkg_root = os.path.join(repo_root, "NeuralTSNE")
            if local_pkg_root not in sys.path:
                sys.path.insert(0, local_pkg_root)
            from NeuralTSNE.TSNE.ParametricTSNE import ParametricTSNE
            from NeuralTSNE.TSNE.Modules import DimensionalityReduction

        X = np.asarray(data, dtype=np.float32)
        n_samples, n_features = X.shape

        # 避免 perplexity >= n_samples 导致计算失败
        perplexity = float(min(30.0, max(2.0, n_samples - 1)))

        # ParametricTSNE 的 P 计算假设每个 batch 都是固定 batch_size；因此对数据做 padding
        # 以确保总样本数是 batch_size 的整数倍（不改变最终返回的样本数）。
        batch_size = int(min(1024, max(2, n_samples)))
        remainder = n_samples % batch_size
        if remainder != 0:
            pad = batch_size - remainder
            X_train = np.concatenate([X, X[:pad]], axis=0)
        else:
            X_train = X

        tsne = ParametricTSNE(
            loss_fn="kl_divergence",
            n_components=n_components,
            perplexity=perplexity,
            batch_size=batch_size,
            early_exaggeration_epochs=0,
            early_exaggeration_value=12,
            max_iterations=250,
            features=n_features,
            multipliers=[0.75, 0.75, 0.75],
            n_jobs=1,
            force_cpu=False,
        )

        is_gpu = tsne.device == torch.device("cuda:0")
        trainer = L.Trainer(
            accelerator="gpu" if is_gpu else "cpu",
            devices=1,
            max_epochs=tsne.max_iterations,
            log_every_n_steps=1,
            enable_checkpointing=False,
            logger=False,
        )

        classifier = DimensionalityReduction(tsne, shuffle=False, lr=1e-3)

        train_dataset = TensorDataset(torch.from_numpy(X_train))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,
        )

        trainer.fit(classifier, train_loader)

        with torch.no_grad():
            emb = tsne.model(torch.from_numpy(X).to(tsne.device)).detach().cpu().numpy()

        reduce_data = emb.astype(np.float32)
        return reduce_data
