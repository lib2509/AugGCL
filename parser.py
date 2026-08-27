import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='AugGCL / LightGCL-improved parameters')

    # Reproducibility
    parser.add_argument('--seed', default=2024, type=int, help='random seed for Python, NumPy and PyTorch')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='enable best-effort deterministic behavior; may reduce speed',
    )

    # Training
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='legacy learning-rate decay argument')
    parser.add_argument('--batch', default=256, type=int, help='number of users per evaluation batch')
    parser.add_argument('--inter_batch', default=4096, type=int, help='interaction training batch size')
    parser.add_argument('--note', default=None, type=str, help='optional run note stored in the metrics CSV')
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of contrastive loss in full mode')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='L2 regularization weight')
    parser.add_argument('--epoch', default=100, type=int, help='number of training epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding dimension')
    parser.add_argument('--q', default=5, type=int, help='rank of the optional SVD view')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of GNN layers')
    parser.add_argument('--data', default='yelp', type=str, help='dataset directory name under data/')
    parser.add_argument('--dropout', default=0.0, type=float, help='View 1 edge-dropout rate during training')
    parser.add_argument('--temp', default=0.2, type=float, help='contrastive-loss temperature')
    parser.add_argument('--cuda', default='0', type=str, help='GPU index; CPU is used if CUDA is unavailable')
    parser.add_argument('--test_every', default=3, type=int, help='evaluate every N epochs; set 0 to disable periodic tests')
    parser.add_argument('--save_every', default=50, type=int, help='save checkpoints every N epochs; set 0 to disable')

    # Four ablation experiments
    parser.add_argument(
        '--ablation',
        default='full',
        choices=['full', 'view1_only', 'view2_only', 'fusion_no_cl'],
        help=(
            'full: BPR(E)+CL(E,G); view1_only: BPR(E); '
            'view2_only: BPR(G) with independent View 2; '
            'fusion_no_cl: BPR((E+G)/2) without CL'
        ),
    )

    # View 2
    parser.add_argument(
        '--view2',
        default='graphaug',
        choices=['graphaug', 'svd'],
        help='GraphAug (AugGCL) or the original LightGCL SVD auxiliary view',
    )
    parser.add_argument(
        '--graphaug_eval',
        default='deterministic',
        choices=['deterministic', 'sample'],
        help='deterministic uses p and p>xi at evaluation; sample uses a fresh Gumbel draw',
    )
    parser.add_argument('--tau1', default=0.5, type=float, help='Gumbel-Sigmoid temperature for GraphAug')
    parser.add_argument('--xi', default=0.5, type=float, help='threshold for retaining a sampled 2-hop edge')
    parser.add_argument('--twohop_per_user', default=20, type=int, help='maximum 2-hop candidates retained per user')
    parser.add_argument('--twohop_seed_items', default=4, type=int, help='seed items sampled from each user history')
    parser.add_argument('--twohop_users_per_item', default=4, type=int, help='neighbor users sampled per seed item')
    parser.add_argument('--twohop_item_sample', default=50, type=int, help='items sampled from each neighbor user')
    parser.add_argument('--aug_hidden', default=64, type=int, help='hidden dimension of the edge augmentor MLP')

    return parser.parse_args()


args = parse_args()
