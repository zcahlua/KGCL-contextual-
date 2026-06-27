from __future__ import annotations

import math

import torch
import torch.nn as nn


class ContextualFGGraphEncoder(nn.Module):
    def __init__(
        self,
        atom_fdim: int,
        bond_fdim: int,
        hidden_size: int,
        fg_type_vocab_size: int,
        fg_max_dist: int = 8,
        layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.fg_max_dist = fg_max_dist
        self.layers = layers
        aux_dim = max(4, min(32, hidden_size // 4))
        self.core_embedding = nn.Embedding(2, aux_dim)
        self.dist_embedding = nn.Embedding(fg_max_dist + 2, aux_dim)
        self.type_embedding = nn.Embedding(fg_type_vocab_size + 1, aux_dim)
        input_dim = atom_fdim + (3 * aux_dim)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear((2 * hidden_size) + bond_fdim + 4, hidden_size),
                    nn.SELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(layers)
            ]
        )
        self.update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_size, hidden_size),
                    nn.SELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(layers)])
        self.boundary_null = nn.Parameter(torch.zeros(hidden_size))
        self.output_mlp = nn.Sequential(
            nn.Linear((2 * hidden_size) + aux_dim, hidden_size),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, graph_tensors) -> torch.Tensor:
        node_features = graph_tensors.fg_node_atom_features
        device = node_features.device
        num_instances = len(graph_tensors.fg_node_scope)
        if num_instances == 0:
            return node_features.new_zeros((0, self.hidden_size))

        core_mask = graph_tensors.fg_node_core_mask.long()
        dist = graph_tensors.fg_node_dist_to_core.clamp(min=0, max=self.fg_max_dist + 1)
        fg_type = graph_tensors.fg_node_fg_type.clamp(min=0, max=self.type_embedding.num_embeddings - 1)
        node_input = torch.cat(
            [
                node_features,
                self.core_embedding(core_mask),
                self.dist_embedding(dist),
                self.type_embedding(fg_type),
            ],
            dim=-1,
        )
        states = self.input_proj(node_input)
        states = states.clone()
        states[0] = 0

        edge_index = graph_tensors.fg_edge_index
        edge_features = graph_tensors.fg_edge_features
        for layer_idx in range(self.layers):
            aggregate = torch.zeros_like(states)
            if edge_index.numel() > 0:
                src = edge_index[0]
                dst = edge_index[1]
                edge_aux = torch.stack(
                    [
                        graph_tensors.fg_node_core_mask[src].float(),
                        graph_tensors.fg_node_core_mask[dst].float(),
                        graph_tensors.fg_node_dist_to_core[src].float() / max(self.fg_max_dist, 1),
                        graph_tensors.fg_node_dist_to_core[dst].float() / max(self.fg_max_dist, 1),
                    ],
                    dim=-1,
                ).to(device)
                message_input = torch.cat([states[src], states[dst], edge_features, edge_aux], dim=-1)
                messages = self.message_mlps[layer_idx](message_input)
                aggregate.index_add_(0, dst, messages)
                counts = torch.zeros(states.size(0), 1, device=device)
                counts.index_add_(0, dst, torch.ones(messages.size(0), 1, device=device))
                aggregate = aggregate / counts.clamp_min(1.0)
            update = self.update_mlps[layer_idx](torch.cat([states, aggregate], dim=-1))
            states = self.layer_norms[layer_idx](states + update)
            states = states.clone()
            states[0] = 0

        pooled = []
        for fg_idx, (node_start, node_count) in enumerate(graph_tensors.fg_node_scope):
            fg_states = states[node_start: node_start + node_count]
            fg_core = graph_tensors.fg_node_core_mask[node_start: node_start + node_count]
            if fg_states.numel() == 0:
                core_vec = torch.zeros(self.hidden_size, device=device)
                boundary_vec = self.boundary_null
                type_id = torch.tensor(0, device=device, dtype=torch.long)
            else:
                core_vec = fg_states[fg_core].mean(dim=0)
                boundary_states = fg_states[~fg_core]
                boundary_vec = boundary_states.mean(dim=0) if boundary_states.numel() else self.boundary_null
                type_id = graph_tensors.fg_node_fg_type[node_start]
            type_vec = self.type_embedding(type_id.clamp(max=self.type_embedding.num_embeddings - 1))
            pooled.append(self.output_mlp(torch.cat([core_vec, boundary_vec, type_vec], dim=-1)))
        return torch.stack(pooled, dim=0)


class KGContextFusion(nn.Module):
    def __init__(
        self,
        kg_dim: int,
        ctx_dim: int,
        chem_descriptor_dim: int,
        out_dim: int,
        use_kg_fusion: bool = True,
        freeze_kg_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.use_kg_fusion = use_kg_fusion
        self.kg_proj = nn.Linear(kg_dim, out_dim)
        self.ctx_proj = nn.Linear(ctx_dim, out_dim)
        self.gate = nn.Linear(kg_dim + ctx_dim + chem_descriptor_dim, out_dim)
        if freeze_kg_embeddings:
            for param in self.kg_proj.parameters():
                param.requires_grad = False

    def forward(
        self,
        kg_embeddings: torch.Tensor,
        contextual_embeddings: torch.Tensor,
        chem_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        if contextual_embeddings.size(0) == 0:
            return contextual_embeddings.new_zeros((0, self.ctx_proj.out_features))
        ctx_hat = self.ctx_proj(contextual_embeddings)
        if not self.use_kg_fusion:
            return ctx_hat
        kg_hat = self.kg_proj(kg_embeddings)
        gamma = torch.sigmoid(self.gate(torch.cat([kg_embeddings, contextual_embeddings, chem_descriptors], dim=-1)))
        return gamma * ctx_hat + (1.0 - gamma) * kg_hat


class AtomFGAttention(nn.Module):
    def __init__(
        self,
        atom_fdim: int,
        fg_dim: int,
        attn_dim: int,
        fg_max_dist: int = 8,
        use_membership_bias: bool = True,
        use_distance_bias: bool = True,
        use_null_token: bool = True,
    ) -> None:
        super().__init__()
        self.atom_fdim = atom_fdim
        self.fg_dim = fg_dim
        self.attn_dim = attn_dim
        self.fg_max_dist = fg_max_dist
        self.use_membership_bias = use_membership_bias
        self.use_distance_bias = use_distance_bias
        self.use_null_token = use_null_token
        self.query = nn.Linear(atom_fdim, attn_dim, bias=False)
        self.key = nn.Linear(fg_dim, attn_dim, bias=False)
        self.value = nn.Linear(fg_dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, atom_fdim, bias=False)
        self.layer_norm = nn.LayerNorm(atom_fdim)
        self.null_token = nn.Parameter(torch.zeros(fg_dim))
        self.beta_in = nn.Parameter(torch.tensor(0.0))
        self.beta_out = nn.Parameter(torch.tensor(0.0))
        self.beta_null = nn.Parameter(torch.tensor(0.0))
        self.distance_bias = nn.Embedding(fg_max_dist + 2, 1)
        self.delta_null = nn.Parameter(torch.tensor(0.0))

    def forward(self, atom_features: torch.Tensor, fg_embeddings: torch.Tensor, graph_tensors, atom_scope):
        enhanced = atom_features.clone()
        fg_context = atom_features.new_zeros((atom_features.size(0), self.attn_dim))
        attention_weights = []
        device = atom_features.device

        for mol_idx, (atom_start, atom_count) in enumerate(atom_scope):
            atoms = atom_features[atom_start: atom_start + atom_count]
            fg_start, fg_count = graph_tensors.fg_scope[mol_idx]
            real_fgs = fg_embeddings[fg_start: fg_start + fg_count]
            include_null = self.use_null_token or fg_count == 0
            groups = real_fgs
            if include_null:
                groups = torch.cat([groups, self.null_token.view(1, -1)], dim=0)
            if groups.size(0) == 0:
                enhanced[atom_start: atom_start + atom_count] = atoms
                attention_weights.append(atoms.new_zeros((atom_count, 0)))
                continue

            scores = self.query(atoms).matmul(self.key(groups).t()) / math.sqrt(self.attn_dim)
            if fg_count > 0:
                membership = graph_tensors.atom_fg_membership[
                    atom_start: atom_start + atom_count,
                    fg_start: fg_start + fg_count,
                ]
                dist = graph_tensors.atom_fg_dist[
                    atom_start: atom_start + atom_count,
                    fg_start: fg_start + fg_count,
                ].clamp(min=0, max=self.fg_max_dist + 1)
            else:
                membership = torch.zeros((atom_count, 0), dtype=torch.bool, device=device)
                dist = torch.zeros((atom_count, 0), dtype=torch.long, device=device)

            if include_null:
                membership = torch.cat(
                    [membership, torch.zeros((atom_count, 1), dtype=torch.bool, device=device)],
                    dim=1,
                )
                dist = torch.cat(
                    [
                        dist,
                        torch.full((atom_count, 1), self.fg_max_dist + 1, dtype=torch.long, device=device),
                    ],
                    dim=1,
                )

            if self.use_membership_bias:
                membership_bias = torch.where(membership, self.beta_in, self.beta_out)
                if include_null:
                    membership_bias[:, -1] = self.beta_null
                scores = scores + membership_bias
            if self.use_distance_bias:
                dist_bias = self.distance_bias(dist).squeeze(-1)
                if include_null:
                    dist_bias[:, -1] = self.delta_null
                scores = scores + dist_bias

            weights = torch.softmax(scores, dim=-1)
            context = weights.matmul(self.value(groups))
            enhanced[atom_start: atom_start + atom_count] = self.layer_norm(atoms + self.output(context))
            fg_context[atom_start: atom_start + atom_count] = context
            attention_weights.append(weights)

        enhanced[0] = 0
        fg_context[0] = 0
        return enhanced, fg_context, attention_weights
