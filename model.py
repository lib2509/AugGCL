import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import sparse_dropout


ABLATION_MODES = ('full', 'view1_only', 'view2_only', 'fusion_no_cl')


class LightGCL(nn.Module):
    """AugGCL / LightGCL-improved with explicit ablation modes.

    Modes
    -----
    full:
        Recommendation backbone = View 1.
        Objective = BPR(View 1) + lambda1 * CL(View 1, View 2) + L2.

    view1_only:
        Recommendation backbone = View 1.
        Objective = BPR(View 1) + L2. View 2 is not constructed.

    view2_only:
        Recommendation backbone = an independent View 2 encoder.
        Objective = BPR(View 2) + L2. No contrastive loss.
        For GraphAug this uses recursive propagation G_l = A' G_{l-1}.

    fusion_no_cl:
        Recommendation backbone = 0.5 * (View 1 + View 2).
        Objective = BPR(Fusion) + L2. No contrastive loss.
    """

    def __init__(
        self,
        n_u,
        n_i,
        d,
        train_csr,
        adj_norm,
        l,
        temp,
        lambda_1,
        lambda_2,
        dropout,
        batch_user,
        device,
        ablation='full',
        # view2 options
        view2_mode='graphaug',
        graphaug_eval='deterministic',
        # svd-view parameters (kept for backward compatibility)
        u_mul_s=None,
        v_mul_s=None,
        ut=None,
        vt=None,
        # graphaug-view parameters
        obs_u=None,
        obs_i=None,
        twohop_u=None,
        twohop_i=None,
        tau1=0.5,
        xi=0.5,
        aug_hidden=None,
    ):
        super().__init__()

        if ablation not in ABLATION_MODES:
            raise ValueError(f'Unknown ablation mode: {ablation}. Expected one of {ABLATION_MODES}.')
        if graphaug_eval not in ('deterministic', 'sample'):
            raise ValueError("graphaug_eval must be either 'deterministic' or 'sample'.")

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.batch_user = batch_user
        self.device = torch.device(device)
        self.ablation = ablation
        self.view2_mode = view2_mode
        self.graphaug_eval = graphaug_eval

        # SVD-view parameters.
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        # GraphAug edge lists.
        self.obs_u = obs_u
        self.obs_i = obs_i
        self.twohop_u = twohop_u
        self.twohop_i = twohop_i
        self.tau1 = tau1
        self.xi = xi

        # Do not create an inactive augmentor for View-1-only or SVD runs.
        self.uses_graphaug = self.view2_mode == 'graphaug' and self.ablation != 'view1_only'
        if self.uses_graphaug:
            hidden = aug_hidden if aug_hidden is not None else d
            self.aug_mlp = nn.Sequential(
                nn.Linear(2 * d, hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden, 1),
            )
        else:
            self.aug_mlp = None

        self.E_u_list = None
        self.E_i_list = None
        self.G_u_list = None
        self.G_i_list = None
        self.E_u = None
        self.E_i = None
        self.G_u = None
        self.G_i = None
        self.final_u = None
        self.final_i = None
        self.last_graphaug_stats = None

    # ------------------------------------------------------------------
    # Public forward / evaluation API
    # ------------------------------------------------------------------
    def forward(self, uids, iids, pos, neg, test=False):
        if test:
            if self.final_u is None or self.final_i is None:
                self.refresh_eval_embeddings()
            return self._recommend(uids)

        self._compute_embeddings(training=True)

        # BPR is always applied to the active recommendation representation.
        u_emb = self.final_u[uids]
        pos_emb = self.final_i[pos]
        neg_emb = self.final_i[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_r = -F.logsigmoid(pos_scores - neg_scores).mean()

        if self.ablation == 'full':
            loss_s_raw = self._contrastive_loss(uids, iids)
            loss_s_weighted = self.lambda_1 * loss_s_raw
        else:
            loss_s_weighted = torch.zeros((), device=self.final_u.device)

        loss_reg = self._regularization_loss()
        loss = loss_r + loss_s_weighted + loss_reg
        return loss, loss_r, loss_s_weighted

    @torch.no_grad()
    def refresh_eval_embeddings(self):
        """Recompute deterministic evaluation embeddings after optimizer updates.

        View 1 is evaluated without edge dropout. GraphAug uses p directly and
        thresholds p at xi by default. Set --graphaug_eval sample to retain a
        stochastic Gumbel sample during evaluation.
        """
        was_training = self.training
        self.eval()
        self._compute_embeddings(training=False)
        if was_training:
            self.train()
        return self.final_u, self.final_i

    def _recommend(self, uids):
        preds = self.final_u[uids] @ self.final_i.T
        mask_np = self.train_csr[uids.detach().cpu().numpy()].toarray()
        mask = torch.as_tensor(mask_np, dtype=preds.dtype, device=preds.device)
        preds = preds.masked_fill(mask.bool(), -1e8)
        return preds.argsort(descending=True)

    # ------------------------------------------------------------------
    # Embedding construction for the four ablations
    # ------------------------------------------------------------------
    def _compute_embeddings(self, training: bool):
        self.last_graphaug_stats = None

        if self.ablation == 'view1_only':
            self._compute_view1(use_dropout=training)
            self.final_u, self.final_i = self.E_u, self.E_i
            return

        if self.ablation == 'view2_only':
            # A genuinely independent View 2. It starts at E_0 and recursively
            # propagates inside the augmented/SVD view.
            self._compute_view2_independent(training=training)
            self.final_u, self.final_i = self.G_u, self.G_i
            return

        # full and fusion_no_cl both need the original graph plus the auxiliary view.
        self._compute_view1(use_dropout=training)
        self._compute_view2_auxiliary(training=training)

        if self.ablation == 'full':
            self.final_u, self.final_i = self.E_u, self.E_i
        elif self.ablation == 'fusion_no_cl':
            self.final_u = 0.5 * (self.E_u + self.G_u)
            self.final_i = 0.5 * (self.E_i + self.G_i)
        else:  # Defensive guard; constructor already validates.
            raise RuntimeError(f'Unhandled ablation mode: {self.ablation}')

    def _compute_view1(self, use_dropout: bool):
        self.E_u_list = [None] * (self.l + 1)
        self.E_i_list = [None] * (self.l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0

        for layer in range(1, self.l + 1):
            adjacency = sparse_dropout(self.adj_norm, self.dropout) if use_dropout else self.adj_norm
            self.E_u_list[layer] = torch.spmm(adjacency, self.E_i_list[layer - 1])
            self.E_i_list[layer] = torch.spmm(adjacency.transpose(0, 1), self.E_u_list[layer - 1])

        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

    def _compute_view2_auxiliary(self, training: bool):
        """LightGCL-style auxiliary view: G_l is generated from E_{l-1}."""
        self.G_u_list = [None] * (self.l + 1)
        self.G_i_list = [None] * (self.l + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        if self.view2_mode == 'svd':
            self._validate_svd_inputs()
            for layer in range(1, self.l + 1):
                self.G_u_list[layer] = self.u_mul_s @ (self.vt @ self.E_i_list[layer - 1])
                self.G_i_list[layer] = self.v_mul_s @ (self.ut @ self.E_u_list[layer - 1])
        else:
            stochastic = training or self.graphaug_eval == 'sample'
            u_all, i_all, v_norm = self._build_graphaug_view2_edges(
                self.E_u.detach(),
                self.E_i.detach(),
                stochastic=stochastic,
                record_stats=not training,
            )
            for layer in range(1, self.l + 1):
                gu, gi = self._propagate_bipartite_edges(
                    self.E_u_list[layer - 1],
                    self.E_i_list[layer - 1],
                    u_all,
                    i_all,
                    v_norm,
                )
                self.G_u_list[layer] = gu
                self.G_i_list[layer] = gi

        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

    def _compute_view2_independent(self, training: bool):
        """Independent View 2 used by the View-2-only ablation.

        GraphAug propagation is recursive:
            G_l = A' G_{l-1}
        rather than depending on View 1 embeddings.
        """
        self.G_u_list = [None] * (self.l + 1)
        self.G_i_list = [None] * (self.l + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        if self.view2_mode == 'svd':
            self._validate_svd_inputs()
            for layer in range(1, self.l + 1):
                self.G_u_list[layer] = self.u_mul_s @ (self.vt @ self.G_i_list[layer - 1])
                self.G_i_list[layer] = self.v_mul_s @ (self.ut @ self.G_u_list[layer - 1])
        else:
            stochastic = training or self.graphaug_eval == 'sample'
            # No View 1 exists in this ablation. Edge probabilities are therefore
            # predicted from the current trainable base embeddings.
            u_all, i_all, v_norm = self._build_graphaug_view2_edges(
                self.E_u_0.detach(),
                self.E_i_0.detach(),
                stochastic=stochastic,
                record_stats=not training,
            )
            for layer in range(1, self.l + 1):
                gu, gi = self._propagate_bipartite_edges(
                    self.G_u_list[layer - 1],
                    self.G_i_list[layer - 1],
                    u_all,
                    i_all,
                    v_norm,
                )
                self.G_u_list[layer] = gu
                self.G_i_list[layer] = gi

        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def _contrastive_loss(self, uids, iids):
        """Stable equivalent of the repository's cross-view contrastive loss."""
        user_logits = self.G_u[uids] @ self.E_u.T / self.temp
        item_logits = self.G_i[iids] @ self.E_i.T / self.temp
        neg_score = torch.logsumexp(user_logits, dim=1).mean()
        neg_score = neg_score + torch.logsumexp(item_logits, dim=1).mean()

        user_pos = (self.G_u[uids] * self.E_u[uids]).sum(1) / self.temp
        item_pos = (self.G_i[iids] * self.E_i[iids]).sum(1) / self.temp
        pos_score = torch.clamp(user_pos, -5.0, 5.0).mean()
        pos_score = pos_score + torch.clamp(item_pos, -5.0, 5.0).mean()
        return -pos_score + neg_score

    def _regularization_loss(self):
        """Regularize only parameters that participate in the active ablation."""
        squared_norm = self.E_u_0.norm(2).square() + self.E_i_0.norm(2).square()
        if self.uses_graphaug and self.aug_mlp is not None:
            for parameter in self.aug_mlp.parameters():
                squared_norm = squared_norm + parameter.norm(2).square()
        return self.lambda_2 * squared_norm

    # ------------------------------------------------------------------
    # GraphAug
    # ------------------------------------------------------------------
    def _build_graphaug_view2_edges(
        self,
        E_u_detached: torch.Tensor,
        E_i_detached: torch.Tensor,
        stochastic: bool,
        record_stats: bool = False,
    ):
        """Construct normalized GraphAug edge lists.

        A' = observed edges (weight 1) union sampled 2-hop candidates.

        During training, candidates use Gumbel-Sigmoid. During deterministic
        evaluation, ``soft = p`` and the threshold is applied directly to p.
        Edge-list propagation is used so gradients can reach edge weights and
        the augmentor MLP.
        """
        if self.aug_mlp is None:
            raise RuntimeError('GraphAug was requested, but aug_mlp is inactive.')
        if self.obs_u is None or self.obs_i is None:
            raise ValueError('obs_u/obs_i must be provided for GraphAug.')
        if self.twohop_u is None or self.twohop_i is None:
            raise ValueError('twohop_u/twohop_i must be provided for GraphAug.')

        candidate_count = int(self.twohop_u.numel())
        if candidate_count > 0:
            hu = E_u_detached[self.twohop_u]
            hv = E_i_detached[self.twohop_i]
            logits = self.aug_mlp(torch.cat([hu, hv], dim=1)).squeeze(-1)
            p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

            if stochastic:
                uniform = torch.rand_like(p)
                gumbel = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
                logit_p = torch.log(p) - torch.log1p(-p)
                soft = torch.sigmoid((logit_p + gumbel) / self.tau1)
            else:
                soft = p

            keep = (soft > self.xi).to(soft.dtype)
            v_twohop = soft * keep
        else:
            p = torch.empty(0, dtype=self.E_u_0.dtype, device=self.E_u_0.device)
            soft = p
            keep = p
            v_twohop = p

        u_all = torch.cat([self.obs_u, self.twohop_u], dim=0)
        i_all = torch.cat([self.obs_i, self.twohop_i], dim=0)
        observed_weights = torch.ones(self.obs_u.numel(), dtype=self.E_u_0.dtype, device=self.E_u_0.device)
        v_all = torch.cat([observed_weights, v_twohop], dim=0)

        n_u = self.E_u_0.shape[0]
        n_i = self.E_i_0.shape[0]
        deg_u = torch.zeros(n_u, device=v_all.device, dtype=v_all.dtype)
        deg_i = torch.zeros(n_i, device=v_all.device, dtype=v_all.dtype)
        deg_u.index_add_(0, u_all, v_all)
        deg_i.index_add_(0, i_all, v_all)
        norm = torch.sqrt((deg_u[u_all] * deg_i[i_all]).clamp_min(1e-12))
        v_norm = v_all / norm

        if record_stats:
            kept_count = int(keep.sum().item()) if candidate_count else 0
            self.last_graphaug_stats = {
                'candidate_edges': candidate_count,
                'kept_edges': kept_count,
                'keep_ratio': kept_count / max(candidate_count, 1),
                'mean_probability': float(p.mean().item()) if candidate_count else 0.0,
                'mean_soft_weight': float(soft.mean().item()) if candidate_count else 0.0,
                'added_weight_sum': float(v_twohop.sum().item()) if candidate_count else 0.0,
            }

        return u_all, i_all, v_norm

    @staticmethod
    def _propagate_bipartite_edges(
        E_u_prev: torch.Tensor,
        E_i_prev: torch.Tensor,
        u_idx: torch.Tensor,
        i_idx: torch.Tensor,
        w: torch.Tensor,
    ):
        n_u, d = E_u_prev.shape
        n_i = E_i_prev.shape[0]

        msg_u = E_i_prev[i_idx] * w.unsqueeze(1)
        E_u_next = torch.zeros((n_u, d), device=E_u_prev.device, dtype=E_u_prev.dtype)
        E_u_next.index_add_(0, u_idx, msg_u)

        msg_i = E_u_prev[u_idx] * w.unsqueeze(1)
        E_i_next = torch.zeros((n_i, d), device=E_i_prev.device, dtype=E_i_prev.dtype)
        E_i_next.index_add_(0, i_idx, msg_i)
        return E_u_next, E_i_next

    def _validate_svd_inputs(self):
        if any(value is None for value in (self.u_mul_s, self.v_mul_s, self.ut, self.vt)):
            raise ValueError('SVD View 2 requires u_mul_s, v_mul_s, ut and vt.')
