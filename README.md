# AugGCL (LightGCL-improved)

**AugGCL** is a PyTorch implementation of a graph-augmentation-based contrastive learning method for recommendation, developed from the original [LightGCL](https://openreview.net/forum?id=FKXVK9dyMM) codebase.

The main modification is the construction of **View 2**. Instead of relying only on the low-rank SVD view used by LightGCL, AugGCL introduces a learnable graph augmentation module that samples candidate user-item connections, predicts their usefulness, and constructs an augmented interaction graph for contrastive learning.

The repository also retains the original **SVD auxiliary view** as an optional compatibility mode.

---

## 1. Model overview

AugGCL uses two graph views.

### View 1 — Original interaction graph

View 1 applies LightGCN-style propagation on the normalized user-item interaction graph $G$:

- learnable user and item embeddings are initialized as $E_U^{(0)}$ and $E_I^{(0)}$;
- embeddings are propagated through the original bipartite graph;
- embeddings from all propagation layers are summed to obtain the final View 1 representations $E_U$ and $E_I$;
- Bayesian Personalized Ranking (**BPR**) is applied to the active recommendation representation.

### View 2 — Learnable graph augmentation

With `--view2 graphaug`, AugGCL constructs the auxiliary view as follows:

1. **Candidate sampling**  
   For each user, candidate items are collected through the interaction chain

   ```text
   user -> interacted item -> neighboring user -> candidate item
   ```

   Items already observed by the target user are removed from the candidate set.

2. **Edge probability prediction**  
   For each candidate pair $(u, i)$, an MLP receives the concatenated user and item embeddings and predicts

   $$
   p_{ui} = \sigma(\mathrm{MLP}([e_u \Vert e_i])).
   $$

3. **Gumbel-Sigmoid relaxation**  
   During training, the probability is transformed into a differentiable stochastic soft edge value using Gumbel-Sigmoid with temperature `tau1`.

4. **Thresholding**  
   A candidate edge is retained when its soft value is greater than `xi`. Retained candidate edges keep their soft values as weights, while observed interactions have weight 1.

5. **Weighted graph normalization and propagation**  
   The observed and retained candidate edges form the augmented graph $G'$. Edge weights are symmetrically normalized before graph propagation.

6. **Cross-view contrastive learning**  
   In the full AugGCL setting, contrastive loss aligns the View 1 and View 2 user/item representations.

For the default `full` mode, the optimization objective is

$$
\mathcal{L}
= \mathcal{L}_{BPR}
+ \lambda_1 \mathcal{L}_{CL}
+ \lambda_2 \mathcal{L}_{2}.
$$

The recommendation backbone in `full` mode remains **View 1**; View 2 is used as the auxiliary contrastive view.

### Evaluation behavior

For GraphAug, evaluation is deterministic by default:

```text
soft = p
keep edge if p > xi
```

Use `--graphaug_eval sample` if a fresh Gumbel sample is desired during evaluation.

---

## 2. Repository structure

```text
LightGCL-improved/
├── data/
│   ├── yelp/
│   ├── gowalla/
│   ├── beer_advocate/
│   ├── last_fm/
│   ├── amazon.zip
│   ├── ml10m.zip
│   └── tmall.zip
├── log/                    # Training metrics and run configurations
├── saved_model/            # Model and optimizer checkpoints
├── old_setting/            # Original LightGCL old-setting implementation
├── main.py                 # Data loading, training, evaluation, logging
├── model.py                # AugGCL model, GraphAug, losses, ablations
├── parser.py               # Command-line arguments
├── utils.py                # Dataset, metrics, sparse utilities
└── README.md
```

`main.py` automatically creates `log/` and `saved_model/` if they do not already exist.

---

## 3. Dataset format

Each dataset directory used by `main.py` must contain at least:

```text
data/<dataset_name>/
├── trnMat.pkl
└── tstMat.pkl
```

Both files are expected to contain SciPy sparse user-item interaction matrices.

Some included datasets also contain `valMat.pkl`, but the current `main.py` does not use a validation matrix.

The large datasets `ML-10M`, `Amazon`, and `Tmall` are provided as compressed files. Unzip the corresponding archive before running experiments on them.

---

## 4. Running environment

The original LightGCL implementation reports the following environment:

```text
Python 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
```

The current AugGCL code additionally uses **pandas** and **SciPy**.

A minimal environment can be installed with:

```bash
pip install torch numpy scipy pandas tqdm
```

CUDA is optional. If CUDA is unavailable, the code automatically falls back to CPU execution.

---

## 5. Running AugGCL

### Full AugGCL on Yelp

```bash
python main.py --data yelp --view2 graphaug --ablation full
```

Since `graphaug` and `full` are the default values, the equivalent shorter command is:

```bash
python main.py --data yelp
```

### Select a GPU

```bash
python main.py --data yelp --cuda 0
```

### Reproducible run

```bash
python main.py \
  --data yelp \
  --view2 graphaug \
  --ablation full \
  --seed 2024 \
  --deterministic
```

`--deterministic` enables best-effort deterministic behavior. Some sparse CUDA operations may still have implementation-dependent limitations.

---

## 6. Ablation modes

The implementation supports four ablation modes.

| Mode | Recommendation representation | Training objective | View 2 |
|---|---|---|---|
| `full` | View 1 | BPR + contrastive loss + L2 | Yes |
| `view1_only` | View 1 | BPR + L2 | No |
| `view2_only` | Independent View 2 | BPR + L2 | Yes |
| `fusion_no_cl` | $(E + G)/2$ | BPR + L2 | Yes |

### Full model

```bash
python main.py --data yelp --view2 graphaug --ablation full
```

### View 1 only

```bash
python main.py --data yelp --view2 graphaug --ablation view1_only
```

In this mode, View 2 and the graph augmentor are not constructed.

### View 2 only

```bash
python main.py --data yelp --view2 graphaug --ablation view2_only
```

For GraphAug, View 2 is independent and recursively propagates on the augmented graph:

```text
G_l = A' G_(l-1)
```

### Fusion without contrastive learning

```bash
python main.py --data yelp --view2 graphaug --ablation fusion_no_cl
```

The recommendation representation is the average of View 1 and View 2 embeddings, without contrastive loss.

---

## 7. Using the original SVD auxiliary view

The code retains the low-rank SVD auxiliary view from LightGCL.

```bash
python main.py --data yelp --view2 svd --ablation full
```

The SVD rank is controlled by:

```bash
--q 5
```

This option is useful for comparison with the LightGCL-style auxiliary view. `--q` is not used by GraphAug.

---

## 8. Main configurable arguments

### General training arguments

| Argument | Default | Description |
|---|---:|---|
| `--seed` | `2024` | Random seed for Python, NumPy, and PyTorch |
| `--deterministic` | off | Enable best-effort deterministic execution |
| `--lr` | `1e-3` | Learning rate |
| `--batch` | `256` | Number of users per evaluation batch |
| `--inter_batch` | `4096` | Interaction training batch size |
| `--epoch` | `100` | Number of training epochs |
| `--d` | `64` | Embedding dimension |
| `--gnn_layer` | `2` | Number of graph propagation layers |
| `--lambda1` | `0.2` | Weight of contrastive loss in `full` mode |
| `--lambda2` | `1e-7` | L2 regularization weight |
| `--temp` | `0.2` | Cross-view contrastive temperature |
| `--dropout` | `0.0` | View 1 edge-dropout rate during training |
| `--data` | `yelp` | Dataset directory under `data/` |
| `--cuda` | `0` | CUDA device index |
| `--test_every` | `3` | Evaluate every N epochs; `0` disables periodic evaluation |
| `--save_every` | `50` | Save checkpoints every N epochs; `0` disables periodic checkpoints |
| `--note` | `None` | Optional note stored in the metrics CSV |

`--decay` is retained as a legacy command-line argument but is not currently used by the training loop.

### GraphAug arguments

| Argument | Default | Description |
|---|---:|---|
| `--view2` | `graphaug` | Auxiliary view: `graphaug` or `svd` |
| `--graphaug_eval` | `deterministic` | GraphAug evaluation: `deterministic` or `sample` |
| `--tau1` | `0.5` | Gumbel-Sigmoid temperature |
| `--xi` | `0.5` | Threshold for retaining candidate edges |
| `--twohop_per_user` | `20` | Maximum candidate edges retained in the candidate pool per user |
| `--twohop_seed_items` | `4` | Number of seed items sampled from each user's history |
| `--twohop_users_per_item` | `4` | Number of neighboring users sampled per seed item |
| `--twohop_item_sample` | `50` | Maximum items sampled from each neighboring user |
| `--aug_hidden` | `64` | Hidden dimension of the edge-scoring MLP |

### SVD argument

| Argument | Default | Description |
|---|---:|---|
| `--q` | `5` | Rank of the SVD auxiliary view |

---

## 9. Example GraphAug experiments

### Change the Gumbel-Sigmoid temperature

```bash
python main.py --data yelp --tau1 1.0
```

### Change the edge-retention threshold

```bash
python main.py --data yelp --xi 0.25
```

### Increase the candidate pool

```bash
python main.py \
  --data yelp \
  --twohop_per_user 20 \
  --twohop_seed_items 10 \
  --twohop_users_per_item 10 \
  --twohop_item_sample 100
```

### Stochastic GraphAug evaluation

```bash
python main.py --data yelp --graphaug_eval sample
```

---

## 10. Evaluation metrics

The current implementation reports:

- Recall@20
- NDCG@20
- Recall@40
- NDCG@40

Training interactions are masked before ranking, so already-observed items are excluded from recommendation results.

By default, evaluation is performed every 3 epochs and once more after the final training epoch.

---

## 11. Output files

Each run receives a tag of the form:

```text
<dataset>_<ablation>_<view2>_seed<seed>_<timestamp>
```

For example:

```text
yelp_full_graphaug_seed2024_20260827-153000
```

### Metrics

Training losses and evaluation metrics are saved to:

```text
log/result_<run_tag>.csv
```

The CSV contains training/test records including:

```text
epoch
split
loss
loss_r
loss_s
recall@20
ndcg@20
recall@40
ndcg@40
data
ablation
view2
seed
note
```

### Configuration

The complete command-line configuration is saved to:

```text
log/config_<run_tag>.json
```

### Checkpoints

Final model and optimizer states are saved to:

```text
saved_model/<run_tag>_model.pt
saved_model/<run_tag>_optim.pt
```

Periodic checkpoints are also created according to `--save_every`.

---

## 12. GraphAug statistics

During GraphAug evaluation, the implementation can report statistics such as:

```text
candidate_edges
kept_edges
keep_ratio
mean_probability
mean_soft_weight
added_weight_sum
```

These values are useful for analyzing how aggressively the augmentation module modifies the original interaction graph under different `tau1` and `xi` settings.

---

## 13. Relationship to LightGCL

This repository is developed from the official LightGCL implementation:

> Xuheng Cai, Chao Huang, Lianghao Xia, and Xubin Ren.  
> **LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation.**  
> International Conference on Learning Representations (ICLR), 2023.

The central difference is the auxiliary-view construction:

| | LightGCL | AugGCL |
|---|---|---|
| View 1 | Original interaction graph | Original interaction graph |
| View 2 | Low-rank SVD graph view | Learnable augmented graph by default |
| Candidate edges | Not used | Sampled from user-item neighborhood structure |
| Edge scorer | Not used | MLP |
| Differentiable edge sampling | Not used | Gumbel-Sigmoid |
| Edge filtering | Not used | Threshold `xi` |
| Contrastive learning | Cross-view | Cross-view |

The SVD View 2 remains available through `--view2 svd` for controlled comparison.

---

## 14. Citation

If you use the original LightGCL method or this codebase derived from it, please cite the LightGCL paper:

```bibtex
@inproceedings{caisimple,
  title={LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation},
  author={Cai, Xuheng and Huang, Chao and Xia, Lianghao and Ren, Xubin},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```

---

## 15. Acknowledgement

This implementation is based on the official LightGCL codebase and extends it with a learnable graph augmentation mechanism and explicit ablation settings for experimental analysis.
