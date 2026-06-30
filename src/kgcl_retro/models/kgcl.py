from typing import Dict, List, Tuple, Union  # Explanation: imports selected names needed to define the KGCL graph-edit prediction model

import torch  # Explanation: imports torch for define the KGCL graph-edit prediction model
import torch.nn as nn  # Explanation: imports torch.nn as nn for define the KGCL graph-edit prediction model
import torch.nn.functional as F  # Explanation: imports torch.nn.functional as F for define the KGCL graph-edit prediction model
from kgcl_retro.chemistry.apply import apply_edit_to_mol  # Explanation: imports the shared packaged edit-application helper.
from rdkit import Chem  # Explanation: imports selected names needed to define the KGCL graph-edit prediction model
from kgcl_retro.data.collate import get_batch_graphs  # Explanation: imports packaged graph batching for autoregressive prediction.
from kgcl_retro.chemistry.fg_instances import FG_CHEM_DESCRIPTOR_SIZE
from kgcl_retro.chemistry.functional_groups import get_functional_group_asset_metadata
from kgcl_retro.chemistry.graphs import MolGraph, Vocab  # Explanation: imports packaged graph and vocabulary helpers.

from kgcl_retro.models.contextual_fg import AtomFGAttention, ContextualFGGraphEncoder, KGContextFusion
from kgcl_retro.models.encoder import Global_Attention, MPNEncoder  # Explanation: imports packaged D-MPNN encoder and optional global attention.
from kgcl_retro.models.utils import (creat_edits_feats, index_select_ND,  # Explanation: imports packaged tensor utilities for edit scoring.
                                     unbatch_feats)  # Explanation: completes the packaged model utility import list.


def add_fg_config_defaults(config: Dict) -> Dict:
    original_config = dict(config)
    config = dict(config)
    config.setdefault("fg_mode", "legacy")
    if config["fg_mode"] == "contextual_fg":
        config["fg_mode"] = "contextual"
    if config["fg_mode"] not in {"legacy", "contextual", "none"}:
        raise ValueError(f"Unsupported fg_mode: {config['fg_mode']}")
    config.setdefault("fg_context_radius", 1)
    config.setdefault("fg_hidden_size", config["mpn_size"])
    if config["fg_hidden_size"] is None:
        config["fg_hidden_size"] = config["mpn_size"]
    config.setdefault("fg_layers", 2)
    config.setdefault("fg_dropout", config.get("dropout_mpn", 0.15))
    if config["fg_dropout"] is None:
        config["fg_dropout"] = config.get("dropout_mpn", 0.15)
    config.setdefault("fg_attn_dim", config["mpn_size"])
    if config["fg_attn_dim"] is None:
        config["fg_attn_dim"] = config["mpn_size"]
    config.setdefault("fg_max_dist", 8)
    config.setdefault("fg_max_matches_per_pattern", None)
    config.setdefault("fg_use_kg_fusion", True)
    config.setdefault("fg_use_membership_bias", True)
    config.setdefault("fg_use_distance_bias", True)
    config.setdefault("fg_null_token", True)
    projection_present = original_config.get("fg_freeze_kg_projection") is not None
    embeddings_present = original_config.get("fg_freeze_kg_embeddings") is not None
    if projection_present and embeddings_present:
        if bool(original_config["fg_freeze_kg_projection"]) != bool(original_config["fg_freeze_kg_embeddings"]):
            raise ValueError(
                "Conflicting FG freeze flags: fg_freeze_kg_projection and deprecated "
                "fg_freeze_kg_embeddings differ."
            )
    if projection_present:
        config["fg_freeze_kg_projection"] = bool(original_config["fg_freeze_kg_projection"])
    elif embeddings_present:
        config["fg_freeze_kg_projection"] = bool(original_config["fg_freeze_kg_embeddings"])
    else:
        config["fg_freeze_kg_projection"] = False
    config["fg_freeze_kg_embeddings"] = config["fg_freeze_kg_projection"]
    config.setdefault("fg_debug", False)
    config.setdefault("use_rxn_class", False)
    return config


def validate_model_config(config: Dict) -> None:
    if config.get("atom_message", False):
        raise ValueError(
            "atom_message=True is not implemented in KGCL-contextual. "
            "Use atom_message=False. The current encoder is directed-bond-message D-MPNN."
        )


class KGCL(nn.Module):  # Explanation: defines KGCL, main KGCL neural network for graph-edit retrosynthesis
    def __init__(self,  # Explanation: defines __init__, which define the KGCL graph-edit prediction model
                 config: Dict,  # Explanation: continues the current multi-line argument or data structure
                 atom_vocab: Vocab,  # Explanation: continues the current multi-line argument or data structure
                 bond_vocab: Vocab,  # Explanation: continues the current multi-line argument or data structure
                 device: str = 'cpu') -> None:  # Explanation: assigns an intermediate value used by later computation
        """
        Parameters
        ----------
        config: Dict, Model arguments
        atom_vocab: atom and LG edit labels
        bond_vocab: bond edit labels
        device: str, Device to run the model on.
        """
        super(KGCL, self).__init__()  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model

        self.config = add_fg_config_defaults(config)  # Explanation: stores this value on the object for later model operations
        validate_model_config(self.config)
        self.atom_vocab = atom_vocab  # Explanation: stores this value on the object for later model operations
        self.bond_vocab = bond_vocab  # Explanation: stores this value on the object for later model operations
        self.atom_outdim = len(atom_vocab)  # Explanation: stores this value on the object for later model operations
        self.bond_outdim = len(bond_vocab)  # Explanation: stores this value on the object for later model operations
        self.device = device  # Explanation: stores this value on the object for later model operations
        self.last_contextual_fg_diagnostics = {}

        self._build_layers()  # Explanation: uses or updates this object state during computation

    def _build_layers(self) -> None:  # Explanation: defines _build_layers, which define the KGCL graph-edit prediction model
        """Builds the different layers associated with the model."""
        config = self.config  # Explanation: assigns an intermediate value used by later computation
        self.encoder = MPNEncoder(atom_fdim=config['n_atom_feat'],  # Explanation: stores this value on the object for later model operations
                                  bond_fdim=config['n_bond_feat'],  # Explanation: computes an intermediate value for molecular graph editing
                                  hidden_size=config['mpn_size'],  # Explanation: assigns an intermediate value used by later computation
                                  depth=config['depth'],  # Explanation: assigns an intermediate value used by later computation
                                  dropout=config['dropout_mpn'],  # Explanation: assigns an intermediate value used by later computation
                                  atom_message=config['atom_message'])  # Explanation: computes an intermediate value for molecular graph editing

        self.W_vv = nn.Linear(config['mpn_size'],  # Explanation: creates a learned linear projection
                              config['mpn_size'], bias=False)  # Explanation: assigns an intermediate value used by later computation
        nn.init.eye_(self.W_vv.weight)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
        self.W_vc = nn.Linear(config['mpn_size'],  # Explanation: creates a learned linear projection
                              config['mpn_size'], bias=False)  # Explanation: assigns an intermediate value used by later computation

        if config['use_attn']:  # Explanation: checks this condition to choose the next execution path
            self.attn = Global_Attention(  # Explanation: stores this value on the object for later model operations
                d_model=config['mpn_size'], heads=config['n_heads'])  # Explanation: assigns an intermediate value used by later computation

        self.fg_mode = config.get("fg_mode", "legacy")
        if self.fg_mode == "contextual":
            embedding_set = "KGembedding_2" if config.get("use_rxn_class", False) else "KGembedding"
            fg_type_count, kg_dim, _names = get_functional_group_asset_metadata(embedding_set)
            bond_attr_fdim = config.get("n_bond_attr_feat", config["n_bond_feat"] - config["n_atom_feat"])
            if bond_attr_fdim <= 0:
                raise ValueError(
                    "Contextual FG requires a positive raw bond-attribute dimension. "
                    "Check n_bond_feat, n_atom_feat, and atom_message=False."
                )
            self.contextual_fg_encoder = ContextualFGGraphEncoder(
                atom_fdim=config["n_atom_feat"],
                bond_fdim=bond_attr_fdim,
                hidden_size=config["fg_hidden_size"],
                fg_type_vocab_size=fg_type_count,
                fg_max_dist=config["fg_max_dist"],
                layers=config["fg_layers"],
                dropout=config["fg_dropout"],
            )
            self.kg_context_fusion = KGContextFusion(
                kg_dim=kg_dim,
                ctx_dim=config["fg_hidden_size"],
                chem_descriptor_dim=FG_CHEM_DESCRIPTOR_SIZE,
                out_dim=config["fg_hidden_size"],
                use_kg_fusion=config["fg_use_kg_fusion"],
                freeze_kg_projection=config["fg_freeze_kg_projection"],
            )
            self.atom_fg_attention = AtomFGAttention(
                atom_fdim=config["n_atom_feat"],
                fg_dim=config["fg_hidden_size"],
                attn_dim=config["fg_attn_dim"],
                fg_max_dist=config["fg_max_dist"],
                use_membership_bias=config["fg_use_membership_bias"],
                use_distance_bias=config["fg_use_distance_bias"],
                use_null_token=config["fg_null_token"],
            )

        self.atom_linear = nn.Sequential(  # Explanation: stores this value on the object for later model operations
            nn.Linear(config['mpn_size'], config['mlp_size']),  # Explanation: creates a learned linear projection
            nn.SELU(),  # Explanation: adds SELU nonlinearity
            nn.Dropout(p=config['dropout_mlp']),  # Explanation: adds dropout regularization
            nn.Linear(config['mlp_size'], self.atom_outdim))  # Explanation: creates a learned linear projection
        self.bond_linear = nn.Sequential(  # Explanation: stores this value on the object for later model operations
            nn.Linear(config['mpn_size'] * 2, config['mlp_size']),  # Explanation: creates a learned linear projection
            nn.SELU(),  # Explanation: adds SELU nonlinearity
            nn.Dropout(p=config['dropout_mlp']),  # Explanation: adds dropout regularization
            nn.Linear(config['mlp_size'], self.bond_outdim))  # Explanation: creates a learned linear projection

        self.graph_linear = nn.Sequential(  # Explanation: stores this value on the object for later model operations
            nn.Linear(config['mpn_size'], config['mlp_size']),  # Explanation: creates a learned linear projection
            nn.SELU(),  # Explanation: adds SELU nonlinearity
            nn.Dropout(p=config['dropout_mlp']),  # Explanation: adds dropout regularization
            nn.Linear(config['mlp_size'], 1))  # Explanation: creates a learned linear projection

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:  # Explanation: defines to_device, which define the KGCL graph-edit prediction model
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if hasattr(tensors, "to") and tensors.__class__.__name__ == "BatchGraphTensors":
            return tensors.to(self.device, non_blocking=True)
        if isinstance(tensors, list) or isinstance(tensors, tuple):  # Explanation: checks this condition to choose the next execution path
            tensors = [tensor.to(self.device, non_blocking=True)  # Explanation: assigns an intermediate value used by later computation
                       for tensor in tensors]  # Explanation: iterates over this collection to process each item
            return tensors  # Explanation: returns this computed result to the caller
        elif isinstance(tensors, torch.Tensor):  # Explanation: checks an alternate condition after the previous branch failed
            return tensors.to(self.device, non_blocking=True)  # Explanation: returns this computed result to the caller
        else:  # Explanation: handles the fallback branch for the preceding condition
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")  # Explanation: raises an error when unsupported input is encountered

    def compute_edit_scores(self, prod_tensors: Tuple[torch.Tensor],  # Explanation: defines compute_edit_scores, which define the KGCL graph-edit prediction model
                            prod_scopes: Tuple[List], prev_atom_hiddens: torch.Tensor = None,  # Explanation: computes an intermediate value for molecular graph editing
                            prev_atom_scope: Tuple[List] = None) :  # Explanation: assigns an intermediate value used by later computation
        """Computes the edit scores given product tensors and scopes.

        Parameters
        ----------
        prod_tensors: Tuple[torch.Tensor]:
            Product tensors
        prod_scopes: Tuple[List]
            Product scopes. Scopes is composed of atom and bond scopes, which
            keep track of atom and bond indices for each molecule in the 2D
            feature list
        prev_atom_hiddens: torch.Tensor, default None,
            Previous hidden state of atoms.
        """
        prod_tensors = self.to_device(prod_tensors)  # Explanation: computes an intermediate value for molecular graph editing
        atom_scope, bond_scope = prod_scopes  # Explanation: computes an intermediate value for molecular graph editing
        if prev_atom_hiddens is None:  # Explanation: checks this condition to choose the next execution path
            n_atoms = prod_tensors[0].size(0)  # Explanation: assigns an intermediate value used by later computation
            prev_atom_hiddens = torch.zeros(  # Explanation: assigns an intermediate value used by later computation
                n_atoms, self.config['mpn_size'], device=self.device)  # Explanation: assigns an intermediate value used by later computation

        if self.config["fg_mode"] == "contextual":
            if not hasattr(prod_tensors, "has_contextual_fg"):
                raise ValueError(
                    "Contextual FG mode requires contextual prepared graph tensors. "
                    "Regenerate data with kgcl-prepare-data --fg_mode contextual."
                )
            contextual_embeddings = self.contextual_fg_encoder(prod_tensors)
            fg_embeddings = self.kg_context_fusion(
                prod_tensors.fg_kg_embeddings,
                contextual_embeddings,
                prod_tensors.fg_chem_descriptors,
            )
            enhanced_atoms, self.last_atom_fg_context, self.last_atom_fg_attention = self.atom_fg_attention(
                prod_tensors.f_atoms,
                fg_embeddings,
                prod_tensors,
                atom_scope,
            )
            if self.config.get("fg_debug", False):
                self.last_contextual_fg_diagnostics = self._contextual_fg_diagnostics(
                    prod_tensors, self.last_atom_fg_attention
                )
            a_feats = self.encoder(
                prod_tensors,
                mask=None,
                atom_features_override=enhanced_atoms,
                bond_attrs=prod_tensors.f_bond_attrs,
            )
        else:
            a_feats = self.encoder(prod_tensors, mask=None)  # Explanation: assigns an intermediate value used by later computation
        if self.config['use_attn']:  # Explanation: checks this condition to choose the next execution path
            feats, mask = creat_edits_feats(a_feats, atom_scope)  # Explanation: assigns an intermediate value used by later computation
            attention_score, feats = self.attn(feats, mask)  # Explanation: assigns an intermediate value used by later computation
            a_feats = unbatch_feats(feats, atom_scope)  # Explanation: assigns an intermediate value used by later computation

        if a_feats.shape[0] != prev_atom_hiddens.shape[0]:  # Explanation: checks this condition to choose the next execution path
            n_atoms = a_feats.shape[0]  # Explanation: assigns an intermediate value used by later computation
            new_ha = torch.zeros(  # Explanation: assigns an intermediate value used by later computation
                n_atoms, self.config['mpn_size'], device=self.device)  # Explanation: assigns an intermediate value used by later computation
            for idx, ((st_n, le_n), (st_p, le_p)) in enumerate(zip(*(atom_scope, prev_atom_scope))):  # Explanation: iterates over this collection to process each item
                new_ha[st_n: st_n + le_p] = prev_atom_hiddens[st_p: st_p + le_p]  # Explanation: assigns an intermediate value used by later computation
            prev_atom_hiddens = new_ha  # Explanation: assigns an intermediate value used by later computation

        assert a_feats.shape == prev_atom_hiddens.shape  # Explanation: checks an invariant expected by the model pipeline
        atom_feats = F.selu(self.W_vv(prev_atom_hiddens) + self.W_vc(a_feats))  # Explanation: computes an intermediate value for molecular graph editing
        prev_atom_hiddens = atom_feats.clone()  # Explanation: assigns an intermediate value used by later computation
        prev_atom_scope = atom_scope  # Explanation: assigns an intermediate value used by later computation

        node_feats = atom_feats.clone()  # Explanation: assigns an intermediate value used by later computation
        bond_starts = index_select_ND(atom_feats, index=prod_tensors[-1][:, 0])  # Explanation: computes an intermediate value for molecular graph editing
        bond_ends = index_select_ND(atom_feats, index=prod_tensors[-1][:, 1])  # Explanation: computes an intermediate value for molecular graph editing
        bond_feats = torch.cat([bond_starts, bond_ends], dim=1)  # Explanation: concatenates tensors along an existing dimension

        graph_vecs = torch.stack(  # Explanation: stacks tensors along a new dimension
            [atom_feats[st: st + le].sum(dim=0) for st, le in atom_scope])  # Explanation: assigns an intermediate value used by later computation

        atom_outs = self.atom_linear(node_feats)  # Explanation: computes an intermediate value for molecular graph editing
        bond_outs = self.bond_linear(bond_feats)  # Explanation: computes an intermediate value for molecular graph editing
        graph_outs = self.graph_linear(graph_vecs)  # Explanation: computes an intermediate value for molecular graph editing

        edit_scores = [torch.cat([bond_outs[st_b: st_b + le_b].flatten(),  # Explanation: concatenates tensors along an existing dimension
                                  atom_outs[st_a: st_a + le_a].flatten(), graph_outs[idx]], dim=-1)  # Explanation: computes an intermediate value for molecular graph editing
                       for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))]  # Explanation: iterates over this collection to process each item

        return edit_scores, prev_atom_hiddens, prev_atom_scope, graph_vecs  # Explanation: returns this computed result to the caller

    def _contextual_fg_diagnostics(self, prod_tensors, attention_weights: list[torch.Tensor]) -> Dict[str, float]:
        fg_counts = [count for _start, count in prod_tensors.fg_scope]
        context_sizes = [count for _start, count in prod_tensors.fg_node_scope]
        entropies = []
        for weights in attention_weights:
            if weights.numel() == 0:
                continue
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1)
            entropies.append(entropy.detach())
        if entropies:
            entropy_values = torch.cat(entropies)
            mean_entropy = float(entropy_values.mean().cpu())
            max_entropy = float(entropy_values.max().cpu())
        else:
            mean_entropy = 0.0
            max_entropy = 0.0
        return {
            "avg_fg_instances_per_mol": float(sum(fg_counts) / max(len(fg_counts), 1)),
            "avg_fg_context_size": float(sum(context_sizes) / max(len(context_sizes), 1)) if context_sizes else 0.0,
            "null_only_molecules": float(sum(1 for count in fg_counts if count == 0)),
            "mean_atom_fg_attention_entropy": mean_entropy,
            "max_atom_fg_attention_entropy": max_entropy,
        }

    def forward(self, prod_seq_inputs: List[Tuple[torch.Tensor, List]]):  # Explanation: defines forward, which define the KGCL graph-edit prediction model
        """
        Forward propagation step.

        Parameters
        ----------
        prod_seq_inputs: List[Tuple[torch.Tensor, List]]
            List of prod_tensors for edit sequence
        """
        max_seq_len = len(prod_seq_inputs)  # Explanation: assigns an intermediate value used by later computation
        assert len(prod_seq_inputs[0]) == 2  # Explanation: checks an invariant expected by the model pipeline

        prev_atom_hiddens = None  # Explanation: assigns an intermediate value used by later computation
        prev_atom_scope = None  # Explanation: assigns an intermediate value used by later computation
        seq_edit_scores = []  # Explanation: computes an intermediate value for molecular graph editing
        batch_graph_outs = []  # Explanation: assigns an intermediate value used by later computation
        for idx in range(max_seq_len):  # Explanation: iterates over this collection to process each item
            prod_tensors, prod_scopes = prod_seq_inputs[idx]  # Explanation: computes an intermediate value for molecular graph editing
            edit_scores, prev_atom_hiddens, prev_atom_scope, graph_outs = self.compute_edit_scores(  # Explanation: computes an intermediate value for molecular graph editing
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
            seq_edit_scores.append(edit_scores)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
            batch_graph_outs.append(graph_outs)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model

        return seq_edit_scores, batch_graph_outs  # Explanation: returns this computed result to the caller

    def predict(self, prod_smi: str, rxn_class: int = None, max_steps: int = 9):  # Explanation: defines predict, which define the KGCL graph-edit prediction model
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        rxn_class: int, default None
            Associated reaction class for the product
        max_steps: int, default 8
            Max number of edit steps allowed
        """
        use_rxn_class = False  # Explanation: assigns an intermediate value used by later computation
        if rxn_class is not None:  # Explanation: checks this condition to choose the next execution path
            use_rxn_class = True  # Explanation: assigns an intermediate value used by later computation

        done = False  # Explanation: assigns an intermediate value used by later computation
        steps = 0  # Explanation: assigns an intermediate value used by later computation
        edits = []  # Explanation: assigns an intermediate value used by later computation
        edits_atom = []  # Explanation: assigns an intermediate value used by later computation
        prev_atom_hiddens = None  # Explanation: assigns an intermediate value used by later computation
        prev_atom_scope = None  # Explanation: assigns an intermediate value used by later computation

        products = Chem.MolFromSmiles(prod_smi)  # Explanation: parses a SMILES string into an RDKit molecule
        Chem.Kekulize(products)  # Explanation: converts aromatic bonds into kekulized form
        prod_graph = MolGraph(mol=Chem.Mol(products),  # Explanation: computes an intermediate value for molecular graph editing
                              rxn_class=rxn_class, use_rxn_class=use_rxn_class,
                              fg_mode=self.config["fg_mode"],
                              fg_context_radius=self.config["fg_context_radius"],
                              fg_max_matches_per_pattern=self.config["fg_max_matches_per_pattern"],
                              fg_max_dist=self.config["fg_max_dist"])  # Explanation: computes an intermediate value for molecular graph editing
        prod_tensors, prod_scopes = get_batch_graphs(  # Explanation: computes an intermediate value for molecular graph editing
            [prod_graph], use_rxn_class=use_rxn_class, fg_mode=self.config["fg_mode"])  # Explanation: assigns an intermediate value used by later computation

        while not done and steps <= max_steps:  # Explanation: continues looping while the edit-generation condition remains true
            if prod_tensors[-1].size() == (1, 0):  # Explanation: checks this condition to choose the next execution path
                edit = 'Terminate'  # Explanation: assigns an intermediate value used by later computation
                edits.append(edit)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                done = True  # Explanation: assigns an intermediate value used by later computation
                break  # Explanation: exits the current loop early

            edit_logits, prev_atom_hiddens, prev_atom_scope, graph_outs = self.compute_edit_scores(  # Explanation: computes an intermediate value for molecular graph editing
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
            idx = torch.argmax(edit_logits[0])  # Explanation: selects the highest-scoring edit index
            val = edit_logits[0][idx]  # Explanation: assigns an intermediate value used by later computation

            max_bond_idx = products.GetNumBonds() * self.bond_outdim  # Explanation: assigns an intermediate value used by later computation

            if idx.item() == len(edit_logits[0]) - 1:  # Explanation: checks this condition to choose the next execution path
                edit = 'Terminate'  # Explanation: assigns an intermediate value used by later computation
                edits.append(edit)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                done = True  # Explanation: assigns an intermediate value used by later computation
                break  # Explanation: exits the current loop early

            elif idx.item() < max_bond_idx:  # Explanation: checks an alternate condition after the previous branch failed
                bond_logits = edit_logits[0][:products.GetNumBonds(  # Explanation: computes an intermediate value for molecular graph editing
                ) * self.bond_outdim]  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                bond_logits = bond_logits.reshape(  # Explanation: computes an intermediate value for molecular graph editing
                    products.GetNumBonds(), self.bond_outdim)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                idx_tensor = torch.where(bond_logits == val)  # Explanation: assigns an intermediate value used by later computation

                idx_tensor = [indices[-1] for indices in idx_tensor]  # Explanation: assigns an intermediate value used by later computation

                bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()  # Explanation: computes an intermediate value for molecular graph editing
                a1 = products.GetBondWithIdx(  # Explanation: assigns an intermediate value used by later computation
                    bond_idx).GetBeginAtom().GetAtomMapNum()  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                a2 = products.GetBondWithIdx(  # Explanation: assigns an intermediate value used by later computation
                    bond_idx).GetEndAtom().GetAtomMapNum()  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model

                a1, a2 = sorted([a1, a2])  # Explanation: assigns an intermediate value used by later computation
                edit_atom = [a1, a2]  # Explanation: computes an intermediate value for molecular graph editing
                edit = self.bond_vocab.get_elem(edit_idx)  # Explanation: assigns an intermediate value used by later computation

            else:  # Explanation: handles the fallback branch for the preceding condition
                atom_logits = edit_logits[0][max_bond_idx:-1]  # Explanation: computes an intermediate value for molecular graph editing

                assert len(atom_logits) == (  # Explanation: checks that atom logits cover every atom/action pair.
                    products.GetNumAtoms() * self.atom_outdim  # Explanation: computes the expected atom edit logit count.
                )  # Explanation: closes the atom logit shape assertion.
                atom_logits = atom_logits.reshape(  # Explanation: computes an intermediate value for molecular graph editing
                    products.GetNumAtoms(), self.atom_outdim)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                idx_tensor = torch.where(atom_logits == val)  # Explanation: assigns an intermediate value used by later computation

                idx_tensor = [indices[-1] for indices in idx_tensor]  # Explanation: assigns an intermediate value used by later computation
                atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()  # Explanation: computes an intermediate value for molecular graph editing

                a1 = products.GetAtomWithIdx(atom_idx).GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
                edit_atom = a1  # Explanation: computes an intermediate value for molecular graph editing
                edit = self.atom_vocab.get_elem(edit_idx)  # Explanation: assigns an intermediate value used by later computation

            try:  # Explanation: starts a protected block for operations that may fail
                products = apply_edit_to_mol(mol=Chem.Mol(  # Explanation: assigns an intermediate value used by later computation
                    products), edit=edit, edit_atom=edit_atom)  # Explanation: assigns an intermediate value used by later computation
                prod_graph = MolGraph(mol=Chem.Mol(  # Explanation: computes an intermediate value for molecular graph editing
                    products),  rxn_class=rxn_class, use_rxn_class=use_rxn_class,
                    fg_mode=self.config["fg_mode"],
                    fg_context_radius=self.config["fg_context_radius"],
                    fg_max_matches_per_pattern=self.config["fg_max_matches_per_pattern"],
                    fg_max_dist=self.config["fg_max_dist"])  # Explanation: assigns an intermediate value used by later computation
                prod_tensors, prod_scopes = get_batch_graphs(  # Explanation: computes an intermediate value for molecular graph editing
                    [prod_graph], use_rxn_class=use_rxn_class, fg_mode=self.config["fg_mode"])  # Explanation: assigns an intermediate value used by later computation

                edits.append(edit)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                edits_atom.append(edit_atom)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                steps += 1  # Explanation: assigns an intermediate value used by later computation

            except:  # Explanation: handles failures from the preceding try block
                steps += 1  # Explanation: assigns an intermediate value used by later computation
                continue  # Explanation: skips the rest of this loop iteration

        return edits, edits_atom  # Explanation: returns this computed result to the caller

    def get_saveables(self) -> Dict:  # Explanation: defines get_saveables, which define the KGCL graph-edit prediction model
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}  # Explanation: assigns an intermediate value used by later computation
        saveables['config'] = self.config  # Explanation: assigns an intermediate value used by later computation
        saveables['atom_vocab'] = self.atom_vocab  # Explanation: assigns an intermediate value used by later computation
        saveables['bond_vocab'] = self.bond_vocab  # Explanation: assigns an intermediate value used by later computation

        return saveables  # Explanation: returns this computed result to the caller
