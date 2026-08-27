import json
import os
import pickle
import random
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm

from model import LightGCL
from parser import args
from utils import TrnData, metrics, scipy_sparse_mat_to_torch_sparse_tensor


def seed_everything(seed: int, deterministic: bool = False):
    """Seed Python, NumPy and PyTorch for reproducible ablation comparisons."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some sparse CUDA kernels do not have deterministic implementations.
            pass


def build_twohop_candidates(
    train_csr_mat,
    train_csc_mat,
    twohop_per_user,
    seed_items,
    users_per_item,
    item_sample,
    rng,
):
    """Sample user-item 2-hop candidate edges for GraphAug."""
    n_u = train_csr_mat.shape[0]
    twohop_u = []
    twohop_i = []

    for user in range(n_u):
        items_u = train_csr_mat.indices[train_csr_mat.indptr[user]:train_csr_mat.indptr[user + 1]]
        if len(items_u) == 0:
            continue

        if len(items_u) <= seed_items:
            seed = items_u
        else:
            seed = rng.choice(items_u, size=seed_items, replace=False)

        candidate_items = set()
        observed_items = set(items_u.tolist())

        for item in seed:
            users_item = train_csc_mat.indices[
                train_csc_mat.indptr[item]:train_csc_mat.indptr[item + 1]
            ]
            if len(users_item) == 0:
                continue

            if len(users_item) <= users_per_item:
                neighbor_users = users_item
            else:
                neighbor_users = rng.choice(users_item, size=users_per_item, replace=False)

            for neighbor_user in neighbor_users:
                neighbor_items = train_csr_mat.indices[
                    train_csr_mat.indptr[neighbor_user]:train_csr_mat.indptr[neighbor_user + 1]
                ]
                if len(neighbor_items) == 0:
                    continue
                if len(neighbor_items) > item_sample:
                    neighbor_items = rng.choice(neighbor_items, size=item_sample, replace=False)
                candidate_items.update(neighbor_items.tolist())

        candidate_items.difference_update(observed_items)
        if not candidate_items:
            continue

        candidate_array = np.asarray(list(candidate_items), dtype=np.int64)
        if len(candidate_array) > twohop_per_user:
            candidate_array = rng.choice(
                candidate_array,
                size=twohop_per_user,
                replace=False,
            )

        twohop_u.append(np.full(len(candidate_array), user, dtype=np.int64))
        twohop_i.append(candidate_array)

    if not twohop_u:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(twohop_u), np.concatenate(twohop_i)


def evaluate(model, train_user_count, batch_user, test_labels, device):
    """Refresh evaluation embeddings once, then compute Recall/NDCG."""
    model.refresh_eval_embeddings()
    test_uids = np.arange(train_user_count)
    batch_count = int(np.ceil(len(test_uids) / batch_user))

    totals = OrderedDict(
        recall20=0.0,
        ndcg20=0.0,
        recall40=0.0,
        ndcg40=0.0,
    )

    for batch_index in tqdm(range(batch_count), desc='Evaluating', leave=False):
        start = batch_index * batch_user
        end = min((batch_index + 1) * batch_user, len(test_uids))
        batch_uids_np = test_uids[start:end]
        batch_uids = torch.as_tensor(batch_uids_np, dtype=torch.long, device=device)
        predictions = model(batch_uids, None, None, None, test=True).cpu().numpy()

        recall20, ndcg20 = metrics(batch_uids_np, predictions, 20, test_labels)
        recall40, ndcg40 = metrics(batch_uids_np, predictions, 40, test_labels)
        totals['recall20'] += recall20
        totals['ndcg20'] += ndcg20
        totals['recall40'] += recall40
        totals['ndcg40'] += ndcg40

    for key in totals:
        totals[key] /= batch_count
    return totals


def make_run_tag():
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    return f'{args.data}_{args.ablation}_{args.view2}_seed{args.seed}_{timestamp}'


def main():
    seed_everything(args.seed, deterministic=args.deterministic)

    device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')
    print(f'Ablation: {args.ablation} | View 2: {args.view2}')

    os.makedirs('log', exist_ok=True)
    os.makedirs('saved_model', exist_ok=True)
    run_tag = make_run_tag()

    data_path = os.path.join('data', args.data)
    with open(os.path.join(data_path, 'trnMat.pkl'), 'rb') as file:
        train_raw = pickle.load(file)
    with open(os.path.join(data_path, 'tstMat.pkl'), 'rb') as file:
        test = pickle.load(file)

    train_raw = train_raw.tocoo()
    test = test.tocoo()
    train_csr = (train_raw != 0).astype(np.float32).tocsr()

    print('Data loaded.')
    print(
        'user_num:', train_raw.shape[0],
        'item_num:', train_raw.shape[1],
        'lambda_1:', args.lambda1,
        'lambda_2:', args.lambda2,
        'temp:', args.temp,
        'q:', args.q,
    )

    # Symmetric normalization of the bipartite adjacency.
    train_norm = train_raw.copy().astype(np.float32)
    row_degree = np.asarray(train_raw.sum(1)).squeeze()
    col_degree = np.asarray(train_raw.sum(0)).squeeze()
    denominator = np.sqrt(
        row_degree[train_norm.row] * col_degree[train_norm.col]
    )
    train_norm.data = train_norm.data / np.maximum(denominator, 1e-12)

    train_data = TrnData(train_norm)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.inter_batch,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train_norm).coalesce().to(device)
    print('Normalized adjacency constructed.')

    needs_view2 = args.ablation != 'view1_only'
    u_mul_s = v_mul_s = svd_u = svd_v = None
    obs_u_t = obs_i_t = twohop_u_t = twohop_i_t = None

    if needs_view2 and args.view2 == 'svd':
        print('Performing low-rank SVD for View 2...')
        svd_u, singular_values, svd_v = torch.svd_lowrank(adj_norm, q=args.q)
        u_mul_s = svd_u @ torch.diag(singular_values)
        v_mul_s = svd_v @ torch.diag(singular_values)
        print('SVD completed.')

    elif needs_view2 and args.view2 == 'graphaug':
        print('Building GraphAug 2-hop candidates...')
        csr_binary = train_csr.copy()
        csr_binary.data = np.ones_like(csr_binary.data)
        csc_binary = csr_binary.tocsc()
        rng = np.random.default_rng(seed=args.seed)

        twohop_u_np, twohop_i_np = build_twohop_candidates(
            csr_binary,
            csc_binary,
            twohop_per_user=args.twohop_per_user,
            seed_items=args.twohop_seed_items,
            users_per_item=args.twohop_users_per_item,
            item_sample=args.twohop_item_sample,
            rng=rng,
        )

        observed = train_csr.tocoo()
        obs_u_np = observed.row.astype(np.int64)
        obs_i_np = observed.col.astype(np.int64)
        print(
            f'Observed edges: {len(obs_u_np):,} | '
            f'2-hop candidate edges: {len(twohop_u_np):,}'
        )

        obs_u_t = torch.as_tensor(obs_u_np, dtype=torch.long, device=device)
        obs_i_t = torch.as_tensor(obs_i_np, dtype=torch.long, device=device)
        twohop_u_t = torch.as_tensor(twohop_u_np, dtype=torch.long, device=device)
        twohop_i_t = torch.as_tensor(twohop_i_np, dtype=torch.long, device=device)
    else:
        print('View 2 construction skipped for view1_only.')

    test_labels = [[] for _ in range(test.shape[0])]
    for row, col in zip(test.row, test.col):
        test_labels[row].append(col)
    print('Test labels processed.')

    model = LightGCL(
        n_u=adj_norm.shape[0],
        n_i=adj_norm.shape[1],
        d=args.d,
        train_csr=train_csr,
        adj_norm=adj_norm,
        l=args.gnn_layer,
        temp=args.temp,
        lambda_1=args.lambda1,
        lambda_2=args.lambda2,
        dropout=args.dropout,
        batch_user=args.batch,
        device=device,
        ablation=args.ablation,
        view2_mode=args.view2,
        graphaug_eval=args.graphaug_eval,
        u_mul_s=u_mul_s,
        v_mul_s=v_mul_s,
        ut=(svd_u.T if svd_u is not None else None),
        vt=(svd_v.T if svd_v is not None else None),
        obs_u=obs_u_t,
        obs_i=obs_i_t,
        twohop_u=twohop_u_t,
        twohop_i=twohop_i_t,
        tau1=args.tau1,
        xi=args.xi,
        aug_hidden=args.aug_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=args.lr)

    history = []
    for epoch in range(args.epoch):
        model.train()
        model.final_u = None
        model.final_i = None

        epoch_loss = 0.0
        epoch_loss_r = 0.0
        epoch_loss_s = 0.0
        train_loader.dataset.neg_sampling()

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            uids, pos, neg = batch
            uids = uids.long().to(device)
            pos = pos.long().to(device)
            neg = neg.long().to(device)
            iids = torch.cat([pos, neg], dim=0)

            optimizer.zero_grad(set_to_none=True)
            loss, loss_r, loss_s = model(uids, iids, pos, neg)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_loss_r += float(loss_r.detach().cpu())
            epoch_loss_s += float(loss_s.detach().cpu())

        batch_count = max(len(train_loader), 1)
        train_record = {
            'epoch': epoch,
            'split': 'train',
            'loss': epoch_loss / batch_count,
            'loss_r': epoch_loss_r / batch_count,
            'loss_s': epoch_loss_s / batch_count,
            'recall@20': np.nan,
            'ndcg@20': np.nan,
            'recall@40': np.nan,
            'ndcg@40': np.nan,
        }
        history.append(train_record)
        print(
            f"Epoch {epoch} | Loss {train_record['loss']:.6f} | "
            f"BPR {train_record['loss_r']:.6f} | "
            f"Weighted CL {train_record['loss_s']:.6f}"
        )

        should_test = args.test_every > 0 and epoch % args.test_every == 0
        if should_test:
            scores = evaluate(model, train_raw.shape[0], args.batch, test_labels, device)
            history.append({
                'epoch': epoch,
                'split': 'test',
                'loss': np.nan,
                'loss_r': np.nan,
                'loss_s': np.nan,
                'recall@20': scores['recall20'],
                'ndcg@20': scores['ndcg20'],
                'recall@40': scores['recall40'],
                'ndcg@40': scores['ndcg40'],
            })
            print('-------------------------------------------')
            print(
                f"Test epoch {epoch} | Recall@20 {scores['recall20']:.6f} | "
                f"NDCG@20 {scores['ndcg20']:.6f} | "
                f"Recall@40 {scores['recall40']:.6f} | "
                f"NDCG@40 {scores['ndcg40']:.6f}"
            )
            if model.last_graphaug_stats is not None:
                stats = model.last_graphaug_stats
                print(
                    'GraphAug eval | '
                    f"kept {stats['kept_edges']:,}/{stats['candidate_edges']:,} "
                    f"({stats['keep_ratio']:.4f}) | "
                    f"mean p {stats['mean_probability']:.4f} | "
                    f"added weight {stats['added_weight_sum']:.2f}"
                )

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join('saved_model', f'{run_tag}_epoch{epoch + 1}_model.pt'),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join('saved_model', f'{run_tag}_epoch{epoch + 1}_optim.pt'),
            )

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    final_scores = evaluate(model, train_raw.shape[0], args.batch, test_labels, device)
    history.append({
        'epoch': 'Final',
        'split': 'test',
        'loss': np.nan,
        'loss_r': np.nan,
        'loss_s': np.nan,
        'recall@20': final_scores['recall20'],
        'ndcg@20': final_scores['ndcg20'],
        'recall@40': final_scores['recall40'],
        'ndcg@40': final_scores['ndcg40'],
    })

    print('-------------------------------------------')
    print(
        f"Final | Recall@20 {final_scores['recall20']:.6f} | "
        f"NDCG@20 {final_scores['ndcg20']:.6f} | "
        f"Recall@40 {final_scores['recall40']:.6f} | "
        f"NDCG@40 {final_scores['ndcg40']:.6f}"
    )
    if model.last_graphaug_stats is not None:
        print('Final GraphAug stats:', model.last_graphaug_stats)

    metrics_path = os.path.join('log', f'result_{run_tag}.csv')
    pd.DataFrame(history).assign(
        data=args.data,
        ablation=args.ablation,
        view2=args.view2,
        seed=args.seed,
        note=args.note,
    ).to_csv(metrics_path, index=False)

    config_path = os.path.join('log', f'config_{run_tag}.json')
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    torch.save(model.state_dict(), os.path.join('saved_model', f'{run_tag}_model.pt'))
    torch.save(optimizer.state_dict(), os.path.join('saved_model', f'{run_tag}_optim.pt'))
    print(f'Metrics saved to: {metrics_path}')
    print(f'Configuration saved to: {config_path}')


if __name__ == '__main__':
    main()
