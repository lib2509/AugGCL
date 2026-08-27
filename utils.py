import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data


def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0.0
    all_ndcg = 0.0

    for index, uid in enumerate(uids):
        prediction = list(predictions[index][:topk])
        label = test_labels[uid]
        if len(label) == 0:
            continue

        hit = 0
        idcg = np.sum([
            np.reciprocal(np.log2(location + 2))
            for location in range(min(topk, len(label)))
        ])
        dcg = 0.0
        for item in label:
            if item in prediction:
                hit += 1
                location = prediction.index(item)
                dcg += np.reciprocal(np.log2(location + 2))

        all_recall += hit / len(label)
        all_ndcg += dcg / idcg
        user_num += 1

    if user_num == 0:
        return 0.0, 0.0
    return all_recall / user_num, all_ndcg / user_num


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    mat = mat.coalesce()
    values = nn.functional.dropout(mat.values(), p=dropout, training=True)
    return torch.sparse_coo_tensor(
        mat.indices(),
        values,
        mat.size(),
        device=values.device,
        dtype=values.dtype,
    ).coalesce()


def spmm(sp, emb, device=None):
    """Sparse-dense multiplication retained for backward compatibility."""
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segments = emb[cols] * sp.values().unsqueeze(1)
    result = torch.zeros(
        (sp.shape[0], emb.shape[1]),
        device=emb.device,
        dtype=emb.dtype,
    )
    result.index_add_(0, rows, col_segments)
    return result


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows), dtype=np.int32)

    def neg_sampling(self):
        for index in range(len(self.rows)):
            user = self.rows[index]
            while True:
                negative_item = np.random.randint(self.dokmat.shape[1])
                if (user, negative_item) not in self.dokmat:
                    break
            self.negs[index] = negative_item

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index], self.cols[index], self.negs[index]
