from dataclasses import dataclass, fields
from typing import Any, List, Tuple  # Explanation: imports selected names needed to collate molecular graphs and edit labels into tensors
import numpy as np  # Explanation: imports numpy as np for collate molecular graphs and edit labels into tensors
import torch  # Explanation: imports torch for collate molecular graphs and edit labels into tensors
from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM  # Explanation: imports packaged atom and bond feature dimensions for label tensor sizing.
from kgcl_retro.chemistry.fg_instances import FG_CHEM_DESCRIPTOR_SIZE
from kgcl_retro.chemistry.functional_groups import get_functional_group_asset_metadata
from kgcl_retro.chemistry.graphs import MolGraph  # Explanation: imports the packaged molecule graph class used by collate functions.


@dataclass
class BatchGraphTensors:
    f_atoms: torch.Tensor
    f_bonds: torch.Tensor
    f_fgs: torch.Tensor
    atom_num: torch.Tensor
    n_mols: torch.Tensor
    a2b: torch.Tensor
    b2a: torch.Tensor
    b2revb: torch.Tensor
    undirected_b2a: torch.Tensor
    f_bond_attrs: torch.Tensor
    fg_node_atom_features: torch.Tensor
    fg_node_core_mask: torch.Tensor
    fg_node_dist_to_core: torch.Tensor
    fg_node_fg_type: torch.Tensor
    fg_edge_index: torch.Tensor
    fg_edge_features: torch.Tensor
    fg_node_scope: list[tuple[int, int]]
    fg_scope: list[tuple[int, int]]
    fg_kg_embeddings: torch.Tensor
    fg_chem_descriptors: torch.Tensor
    fg_type_ids: torch.Tensor
    atom_fg_membership: torch.Tensor
    atom_fg_dist: torch.Tensor
    atom_fg_scope: list[tuple[int, int, int, int]]
    has_contextual_fg: bool
    fg_max_dist: int

    _legacy_field_names = (
        "f_atoms",
        "f_bonds",
        "f_fgs",
        "atom_num",
        "n_mols",
        "a2b",
        "b2a",
        "b2revb",
        "undirected_b2a",
    )

    def as_legacy_tuple(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, name) for name in self._legacy_field_names)

    def __iter__(self):
        return iter(self.as_legacy_tuple())

    def __len__(self) -> int:
        return len(self._legacy_field_names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.as_legacy_tuple()[idx]

    def to(self, device: str | torch.device, non_blocking: bool = True) -> "BatchGraphTensors":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                values[field.name] = value.to(device, non_blocking=non_blocking)
            else:
                values[field.name] = value
        return BatchGraphTensors(**values)

def create_pad_tensor(alist):  # Explanation: defines create_pad_tensor, which pads variable-length index lists
    max_len = max([len(a) for a in alist])  # Explanation: assigns an intermediate value used by later computation
    for a in alist:  # Explanation: iterates over this collection to process each item
        pad_len = max_len - len(a)  # Explanation: assigns an intermediate value used by later computation
        a.extend([0] * pad_len)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
    return torch.tensor(alist, dtype=torch.long)  # Explanation: returns this computed result to the caller


def prepare_edit_labels(graph_batch: List[MolGraph], edits: List[Any], edit_atoms: List[Any], bond_vocab: List, atom_vocab: List) -> torch.tensor:  # Explanation: defines prepare_edit_labels, which creates flattened one-hot edit labels
    """ 
    Prepare edit label including atom edits and bond edits.
    """
    bond_vocab_size = bond_vocab.size()  # Explanation: computes an intermediate value for molecular graph editing
    atom_vocab_size = atom_vocab.size()  # Explanation: computes an intermediate value for molecular graph editing
    edit_labels = []  # Explanation: computes an intermediate value for molecular graph editing

    for prod_graph, edit, edit_atom in zip(graph_batch, edits, edit_atoms):  # Explanation: iterates over this collection to process each item
        bond_label = np.zeros((prod_graph.num_bonds, bond_vocab_size))  # Explanation: computes an intermediate value for molecular graph editing
        atom_label = np.zeros((prod_graph.num_atoms, atom_vocab_size))  # Explanation: computes an intermediate value for molecular graph editing
        stop_label = np.zeros((1,))  # Explanation: assigns an intermediate value used by later computation

        if edit == 'Terminate':  # Explanation: checks this condition to choose the next execution path
            stop_label[0] = 1.0  # Explanation: assigns an intermediate value used by later computation

        elif edit[0] == 'Change Atom' or edit[0] == 'Attaching LG':  # Explanation: checks an alternate condition after the previous branch failed
            a_map = edit_atom  # Explanation: assigns an intermediate value used by later computation
            a_idx = prod_graph.amap_to_idx[a_map]  # Explanation: assigns an intermediate value used by later computation
            edit_idx = atom_vocab.get_index(edit)  # Explanation: computes an intermediate value for molecular graph editing
            atom_label[a_idx][edit_idx] = 1  # Explanation: computes an intermediate value for molecular graph editing

        else:  # Explanation: handles the fallback branch for the preceding condition
            a1, a2 = edit_atom[0], edit_atom[1]  # Explanation: assigns an intermediate value used by later computation
            a_start, a_end = prod_graph.amap_to_idx[a1], prod_graph.amap_to_idx[a2]  # Explanation: assigns an intermediate value used by later computation
            b_idx = prod_graph.mol.GetBondBetweenAtoms(a_start, a_end).GetIdx()  # Explanation: assigns an intermediate value used by later computation
            edit_idx = bond_vocab.get_index(edit)  # Explanation: computes an intermediate value for molecular graph editing
            bond_label[b_idx][edit_idx] = 1  # Explanation: computes an intermediate value for molecular graph editing

        edit_label = np.concatenate(  # Explanation: computes an intermediate value for molecular graph editing
            (bond_label.flatten(), atom_label.flatten(), stop_label.flatten()))  # Explanation: continues a structured literal or expression
        edit_label = torch.from_numpy(edit_label)  # Explanation: computes an intermediate value for molecular graph editing
        edit_labels.append(edit_label)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

    return edit_labels  # Explanation: returns this computed result to the caller


def get_batch_graphs(
    graph_batch: List[MolGraph],
    use_rxn_class: bool = False,
    fg_mode: str | None = None,
) -> Tuple[torch.Tensor, List[Tuple[int]]]:  # Explanation: defines get_batch_graphs, which builds batched molecular graph tensors
    """
    Featurization of a batch of molecules.
    """
    # Start n_atoms and n_bonds at 1 b/c zero padding
    n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)  # Explanation: assigns an intermediate value used by later computation
    n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)  # Explanation: assigns an intermediate value used by later computation
    a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule  # Explanation: assigns an intermediate value used by later computation
    b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule  # Explanation: assigns an intermediate value used by later computation

    # All start with zero padding so that indexing with zero padding returns zeros
    if use_rxn_class:  # Explanation: checks this condition to choose the next execution path
        atom_fdim = ATOM_FDIM + 10  # Explanation: computes an intermediate value for molecular graph editing
    else:  # Explanation: handles the fallback branch for the preceding condition
        atom_fdim = ATOM_FDIM  # Explanation: computes an intermediate value for molecular graph editing
    bond_fdim = atom_fdim + BOND_FDIM  # Explanation: computes an intermediate value for molecular graph editing

    f_atoms = [[0] * atom_fdim]  # atom features  # Explanation: assigns an intermediate value used by later computation
    f_bonds = [[0] * bond_fdim]  # combined atom/bond features  # Explanation: assigns an intermediate value used by later computation
    a2b = [[]]  # mapping from atom index to incoming bond indices  # Explanation: assigns an intermediate value used by later computation
    b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from  # Explanation: assigns an intermediate value used by later computation
    b2revb = [0]  # mapping from bond index to the index of the reverse bond  # Explanation: assigns an intermediate value used by later computation
    undirected_b2a = [[]]  # mapping from the undirected bond index to the beginindex and endindex of the atoms  # Explanation: assigns an intermediate value used by later computation
    n_mols = 0  # Explanation: assigns an intermediate value used by later computation
    if fg_mode is None:
        fg_mode = getattr(graph_batch[0], "fg_mode", "legacy") if graph_batch else "legacy"
    if fg_mode == "contextual_fg":
        fg_mode = "contextual"

    f_fgs = []  # Explanation: assigns an intermediate value used by later computation
    atom_num = []  # Explanation: computes an intermediate value for molecular graph editing
    f_bond_attrs = [[0] * BOND_FDIM]
    fg_node_atom_features = [[0] * atom_fdim]
    fg_node_core_mask = [False]
    fg_node_dist_to_core = [0]
    fg_node_fg_type = [0]
    fg_edge_index = []
    fg_edge_features = []
    fg_node_scope = []
    fg_scope = []
    fg_kg_embeddings = []
    fg_chem_descriptors = []
    fg_type_ids = []
    atom_fg_scope = []
    fg_max_dist = max([getattr(graph, "fg_max_dist", 8) for graph in graph_batch], default=8)
    atom_fg_membership_rows: list[list[bool]] = [[False]]
    atom_fg_dist_rows: list[list[int]] = [[fg_max_dist + 1]]
    global_fg_count = 0
    global_fg_node_count = 1

    for mol_graph in graph_batch:  # Explanation: iterates over this collection to process each item

        f_atoms.extend(mol_graph.f_atoms)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        f_bonds.extend(mol_graph.f_bonds)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        f_bond_attrs.extend(getattr(mol_graph, "f_bond_attrs", [[0] * BOND_FDIM for _ in range(mol_graph.n_bonds)]))
        n_mols += 1  # Explanation: assigns an intermediate value used by later computation

        for a in range(mol_graph.n_atoms):  # Explanation: iterates over this collection to process each item
            a2b.append([b + n_bonds for b in mol_graph.a2b[a]])  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

        for b in range(mol_graph.n_bonds):  # Explanation: iterates over this collection to process each item
            b2a.append(n_atoms + mol_graph.b2a[b])  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
            b2revb.append(n_bonds + mol_graph.b2revb[b])  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

        n_undirected_bonds = len(undirected_b2a)  # Explanation: assigns an intermediate value used by later computation
        for bond in mol_graph.mol.GetBonds():  # Explanation: iterates over this collection to process each item
            undirected_b2a.append(sorted([bond.GetBeginAtomIdx() + n_atoms, bond.GetEndAtomIdx() + n_atoms]))  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

        a_scope.append((n_atoms, mol_graph.n_atoms))  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        b_scope.append((n_undirected_bonds, mol_graph.num_bonds))  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        n_atoms += mol_graph.n_atoms  # Explanation: assigns an intermediate value used by later computation
        n_bonds += mol_graph.n_bonds  # Explanation: assigns an intermediate value used by later computation

        f_fgs.extend(mol_graph.f_fgs)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        atom_num.append(mol_graph.n_atoms)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

        if fg_mode == "contextual":
            mol_fg_start = global_fg_count
            mol_fg_count = len(mol_graph.fg_instances)
            fg_scope.append((mol_fg_start, mol_fg_count))
            atom_fg_scope.append((a_scope[-1][0], a_scope[-1][1], mol_fg_start, mol_fg_count))

            for fg_idx, instance in enumerate(mol_graph.fg_instances):
                fg_node_start = global_fg_node_count
                context_atoms = mol_graph.fg_context_atom_indices[fg_idx]
                core_masks = mol_graph.fg_core_masks[fg_idx]
                distances = mol_graph.fg_dist_to_core[fg_idx]
                fg_type_id = mol_graph.fg_type_ids[fg_idx]
                for atom_idx, core_mask, distance in zip(context_atoms, core_masks, distances):
                    fg_node_atom_features.append(mol_graph.f_atoms_raw[atom_idx])
                    fg_node_core_mask.append(bool(core_mask))
                    fg_node_dist_to_core.append(int(distance))
                    fg_node_fg_type.append(int(fg_type_id))
                    global_fg_node_count += 1
                fg_node_scope.append((fg_node_start, len(context_atoms)))

                for src_local, dst_local, bond_features in mol_graph.fg_context_edges[fg_idx]:
                    fg_edge_index.append((fg_node_start + src_local, fg_node_start + dst_local))
                    fg_edge_features.append(bond_features)

                fg_kg_embeddings.append(instance.kg_embedding)
                fg_chem_descriptors.append(instance.chem_descriptors)
                fg_type_ids.append(fg_type_id)
                global_fg_count += 1

            for atom_idx in range(mol_graph.n_atoms):
                atom_fg_membership_rows.append(
                    [False] * mol_fg_start
                    + [bool(value) for value in mol_graph.atom_to_fg_membership[atom_idx]]
                )
                atom_fg_dist_rows.append(
                    [fg_max_dist + 1] * mol_fg_start
                    + [int(value) for value in mol_graph.atom_to_fg_dist[atom_idx]]
                )

    f_atoms = torch.FloatTensor(f_atoms)  # Explanation: assigns an intermediate value used by later computation
    f_bonds = torch.FloatTensor(f_bonds)  # Explanation: assigns an intermediate value used by later computation
    a2b = create_pad_tensor(a2b)  # Explanation: assigns an intermediate value used by later computation
    b2a = torch.LongTensor(b2a)  # Explanation: assigns an intermediate value used by later computation
    b2revb = torch.LongTensor(b2revb)  # Explanation: assigns an intermediate value used by later computation
    undirected_b2a = create_pad_tensor(undirected_b2a)  # Explanation: assigns an intermediate value used by later computation
    if f_fgs:
        f_fgs = torch.FloatTensor(f_fgs)  # Explanation: assigns an intermediate value used by later computation
    elif fg_mode == "contextual":
        _num_fg_types, kg_dim, _names = get_functional_group_asset_metadata(
            "KGembedding_2" if use_rxn_class else "KGembedding"
        )
        f_fgs = torch.zeros((0, kg_dim), dtype=torch.float32)
    else:
        f_fgs = torch.FloatTensor(f_fgs)  # Explanation: preserves the legacy empty tensor shape.
    atom_num = torch.tensor(atom_num)  # Explanation: computes an intermediate value for molecular graph editing
    n_mols = torch.tensor(n_mols)  # Explanation: assigns an intermediate value used by later computation

    if fg_mode == "contextual":
        if fg_kg_embeddings:
            fg_kg_embeddings_tensor = torch.FloatTensor(fg_kg_embeddings)
        else:
            _num_fg_types, kg_dim, _names = get_functional_group_asset_metadata(
                "KGembedding_2" if use_rxn_class else "KGembedding"
            )
            fg_kg_embeddings_tensor = torch.zeros((0, kg_dim), dtype=torch.float32)
        if fg_chem_descriptors:
            fg_chem_descriptors_tensor = torch.FloatTensor(fg_chem_descriptors)
        else:
            fg_chem_descriptors_tensor = torch.zeros((0, FG_CHEM_DESCRIPTOR_SIZE), dtype=torch.float32)

        total_atoms = len(f_atoms)
        total_fgs = len(fg_kg_embeddings)
        padded_membership = torch.zeros((total_atoms, total_fgs), dtype=torch.bool)
        padded_dist = torch.full((total_atoms, total_fgs), fg_max_dist + 1, dtype=torch.long)
        for atom_idx, row in enumerate(atom_fg_membership_rows):
            if total_fgs and row:
                padded_membership[atom_idx, : min(len(row), total_fgs)] = torch.tensor(
                    row[:total_fgs], dtype=torch.bool
                )
        for atom_idx, row in enumerate(atom_fg_dist_rows):
            if total_fgs and row:
                padded_dist[atom_idx, : min(len(row), total_fgs)] = torch.tensor(
                    row[:total_fgs], dtype=torch.long
                )

        if fg_edge_index:
            fg_edge_index_tensor = torch.LongTensor(fg_edge_index).t().contiguous()
            fg_edge_features_tensor = torch.FloatTensor(fg_edge_features)
        else:
            fg_edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
            fg_edge_features_tensor = torch.zeros((0, BOND_FDIM), dtype=torch.float32)

        graph_tensors = BatchGraphTensors(
            f_atoms=f_atoms,
            f_bonds=f_bonds,
            f_fgs=f_fgs,
            atom_num=atom_num,
            n_mols=n_mols,
            a2b=a2b,
            b2a=b2a,
            b2revb=b2revb,
            undirected_b2a=undirected_b2a,
            f_bond_attrs=torch.FloatTensor(f_bond_attrs),
            fg_node_atom_features=torch.FloatTensor(fg_node_atom_features),
            fg_node_core_mask=torch.BoolTensor(fg_node_core_mask),
            fg_node_dist_to_core=torch.LongTensor(fg_node_dist_to_core),
            fg_node_fg_type=torch.LongTensor(fg_node_fg_type),
            fg_edge_index=fg_edge_index_tensor,
            fg_edge_features=fg_edge_features_tensor,
            fg_node_scope=fg_node_scope,
            fg_scope=fg_scope,
            fg_kg_embeddings=fg_kg_embeddings_tensor,
            fg_chem_descriptors=fg_chem_descriptors_tensor,
            fg_type_ids=torch.LongTensor(fg_type_ids),
            atom_fg_membership=padded_membership,
            atom_fg_dist=padded_dist,
            atom_fg_scope=atom_fg_scope,
            has_contextual_fg=True,
            fg_max_dist=fg_max_dist,
        )
        assert graph_tensors.f_bond_attrs.size(0) == graph_tensors.f_bonds.size(0)
        assert graph_tensors.atom_fg_membership.shape == graph_tensors.atom_fg_dist.shape
    else:
        graph_tensors = (f_atoms, f_bonds, f_fgs, atom_num, n_mols, a2b, b2a, b2revb, undirected_b2a)  # Explanation: computes an intermediate value for molecular graph editing
    scopes = (a_scope, b_scope)  # Explanation: assigns an intermediate value used by later computation

    return graph_tensors, scopes  # Explanation: returns this computed result to the caller
    
