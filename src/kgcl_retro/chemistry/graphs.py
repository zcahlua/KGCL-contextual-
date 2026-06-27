from typing import Any, List, Tuple  # Explanation: imports selected names needed to represent reaction graphs and inject functional-group knowledge
from rdkit import Chem  # Explanation: imports selected names needed to represent reaction graphs and inject functional-group knowledge
from kgcl_retro.chemistry.features import get_atom_features, get_bond_features  # Explanation: imports packaged atom and bond featurizers for graph construction.
from kgcl_retro.chemistry.functional_groups import load_functional_group_resources  # Explanation: imports the package-resource loader for KG functional-group embeddings.
from kgcl_retro.chemistry.fg_instances import compute_fg_chem_descriptors, match_fg_instances  # Explanation: imports contextual functional-group occurrence matching.
import torch  # Explanation: imports torch for represent reaction graphs and inject functional-group knowledge
import torch.nn.functional as F  # Explanation: imports torch.nn.functional as F for represent reaction graphs and inject functional-group knowledge
import math  # Explanation: imports math for represent reaction graphs and inject functional-group knowledge


def match_fg(mol, use_rxn_class):  # Explanation: defines match_fg, which matches functional groups and retrieves KG embeddings
    embedding_set = "KGembedding_2" if use_rxn_class else "KGembedding"  # Explanation: selects the embedding table that matches the reaction-class setting.
    resources = load_functional_group_resources(embedding_set)  # Explanation: loads functional-group SMARTS and embeddings from packaged assets.
    fg_emb = []  # Explanation: collects KG embedding vectors for matched functional groups.
    fg_names = []  # Explanation: assigns an intermediate value used by later computation
    for sm in resources.smarts:  # Explanation: checks each functional-group SMARTS pattern against the molecule.
        if mol.HasSubstructMatch(sm):  # Explanation: handles the case where this molecule contains the functional group.
            name = resources.smarts_to_name[sm]  # Explanation: resolves the matched SMARTS pattern back to a functional-group name.
            fg_emb.append(resources.embeddings[name].tolist())  # Explanation: appends the KG embedding vector used for graph feature fusion.
            fg_names.append(name)  # Explanation: records the matched functional-group name for diagnostics.

    return fg_emb,fg_names  # Explanation: returns this computed result to the caller

def attention(query, key, mask=None, dropout=None):  # Explanation: defines attention, which fuses atom features with functional-group embeddings

    pad_rows = query.size(0) - key.size(0)  # Explanation: assigns an intermediate value used by later computation
    if pad_rows > 0:  # Explanation: checks this condition to choose the next execution path
        zero_padding = torch.zeros(pad_rows, key.size(1))  # Explanation: assigns an intermediate value used by later computation
        key_pad = key.clone()  # Explanation: assigns an intermediate value used by later computation
        key = torch.cat((key_pad, zero_padding), dim=0)  # Explanation: concatenates tensors along an existing dimension

    value = key.clone()  # Explanation: assigns an intermediate value used by later computation

    d_k = key.size(-1)  # Explanation: assigns an intermediate value used by later computation
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Explanation: assigns an intermediate value used by later computation

    p_attn = F.softmax(scores, dim=-1)  # Explanation: converts edit logits into probabilities
    if dropout is not None:  # Explanation: checks this condition to choose the next execution path
        p_attn = dropout(p_attn)  # Explanation: assigns an intermediate value used by later computation

    # Res coonect
    a = torch.matmul(p_attn, value)  # Explanation: assigns an intermediate value used by later computation
    out = query + torch.matmul(p_attn, value)  # Explanation: assigns an intermediate value used by later computation
    return out, p_attn  # Explanation: returns this computed result to the caller


class MolGraph:  # Explanation: defines MolGraph, single molecule graph with features and mappings
    """
    'MolGraph' represents the graph structure and featurization of a single molecule.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        rxn_class: int = None,
        use_rxn_class: bool = False,
        fg_mode: str = "legacy",
        fg_context_radius: int = 1,
        fg_max_matches_per_pattern: int | None = None,
        fg_max_dist: int = 8,
    ) -> None:  # Explanation: defines __init__, which represent reaction graphs and inject functional-group knowledge
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.mol = mol  # Explanation: stores this value on the object for later model operations
        self.rxn_class = rxn_class  # Explanation: stores this value on the object for later model operations
        self.use_rxn_class = use_rxn_class  # Explanation: stores this value on the object for later model operations
        self.fg_mode = "contextual" if fg_mode == "contextual_fg" else fg_mode
        self.fg_context_radius = fg_context_radius
        self.fg_max_matches_per_pattern = fg_max_matches_per_pattern
        self.fg_max_dist = fg_max_dist
        if self.fg_mode not in {"legacy", "contextual", "none"}:
            raise ValueError(f"Unsupported fg_mode: {self.fg_mode}")
        self._build_mol()  # Explanation: uses or updates this object state during computation
        self._build_graph()  # Explanation: uses or updates this object state during computation

    def _build_mol(self) -> None:  # Explanation: defines _build_mol, which represent reaction graphs and inject functional-group knowledge
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()  # Explanation: stores this value on the object for later model operations
        self.num_bonds = self.mol.GetNumBonds()  # Explanation: stores this value on the object for later model operations
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()  # Explanation: stores this value on the object for later model operations
                            for atom in self.mol.GetAtoms()}  # Explanation: iterates over this collection to process each item
        self.idx_to_amap = {value: key for key,  # Explanation: stores this value on the object for later model operations
                                           value in self.amap_to_idx.items()}  # Explanation: executes this statement as part of represent reaction graphs and inject functional-group knowledge

    def _empty_contextual_fg_metadata(self) -> None:
        self.fg_instances = []
        self.fg_context_atom_indices = []
        self.fg_context_edges = []
        self.fg_core_masks = []
        self.fg_dist_to_core = []
        self.fg_kg_embeddings = []
        self.fg_type_ids = []
        self.fg_chem_descriptors = []
        self.atom_to_fg_dist = []
        self.atom_to_fg_membership = []

    def _adjacency(self) -> list[list[tuple[int, Any]]]:
        adjacency: list[list[tuple[int, Any]]] = [[] for _ in range(self.n_atoms)]
        for bond in self.mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            adjacency[begin].append((end, bond))
            adjacency[end].append((begin, bond))
        return adjacency

    def _distances_to_core(self, core_atoms: set[int], adjacency: list[list[tuple[int, Any]]]) -> list[int]:
        inf_bucket = self.fg_max_dist + 1
        distances = [inf_bucket] * self.n_atoms
        queue = list(core_atoms)
        for atom_idx in core_atoms:
            distances[atom_idx] = 0

        head = 0
        while head < len(queue):
            atom_idx = queue[head]
            head += 1
            next_dist = distances[atom_idx] + 1
            if next_dist > inf_bucket:
                continue
            for neighbor_idx, _bond in adjacency[atom_idx]:
                if next_dist < distances[neighbor_idx]:
                    distances[neighbor_idx] = next_dist
                    queue.append(neighbor_idx)
        return distances

    def _build_contextual_fg_metadata(self) -> None:
        self._empty_contextual_fg_metadata()
        self.fg_instances = match_fg_instances(
            self.mol,
            use_rxn_class=self.use_rxn_class,
            max_matches_per_pattern=self.fg_max_matches_per_pattern,
        )
        self.f_fgs = [instance.kg_embedding for instance in self.fg_instances]
        self.fg_names = [instance.name for instance in self.fg_instances]
        adjacency = self._adjacency()
        all_atom_fg_dist: list[list[int]] = []
        all_atom_fg_membership: list[list[bool]] = []

        for instance in self.fg_instances:
            core_atoms = set(instance.core_atoms)
            distances = self._distances_to_core(core_atoms, adjacency)
            context_atom_indices = tuple(
                atom_idx
                for atom_idx, dist in enumerate(distances)
                if dist <= self.fg_context_radius
            )
            context_atom_set = set(context_atom_indices)
            context_position = {atom_idx: idx for idx, atom_idx in enumerate(context_atom_indices)}
            context_edges = []
            for atom_idx in context_atom_indices:
                for neighbor_idx, bond in adjacency[atom_idx]:
                    if neighbor_idx in context_atom_set:
                        context_edges.append(
                            (
                                context_position[atom_idx],
                                context_position[neighbor_idx],
                                get_bond_features(bond),
                            )
                        )

            instance.chem_descriptors = compute_fg_chem_descriptors(
                self.mol, instance.core_atoms, context_atom_indices
            )
            self.fg_context_atom_indices.append(context_atom_indices)
            self.fg_context_edges.append(tuple(context_edges))
            self.fg_core_masks.append(tuple(atom_idx in core_atoms for atom_idx in context_atom_indices))
            self.fg_dist_to_core.append(tuple(min(distances[atom_idx], self.fg_max_dist + 1) for atom_idx in context_atom_indices))
            self.fg_kg_embeddings.append(instance.kg_embedding)
            self.fg_type_ids.append(instance.pattern_index + 1)
            self.fg_chem_descriptors.append(instance.chem_descriptors)
            all_atom_fg_dist.append([min(dist, self.fg_max_dist + 1) for dist in distances])
            all_atom_fg_membership.append([atom_idx in core_atoms for atom_idx in range(self.n_atoms)])

        if self.fg_instances:
            self.atom_to_fg_dist = [
                [all_atom_fg_dist[fg_idx][atom_idx] for fg_idx in range(len(self.fg_instances))]
                for atom_idx in range(self.n_atoms)
            ]
            self.atom_to_fg_membership = [
                [all_atom_fg_membership[fg_idx][atom_idx] for fg_idx in range(len(self.fg_instances))]
                for atom_idx in range(self.n_atoms)
            ]
        else:
            self.atom_to_fg_dist = [[] for _ in range(self.n_atoms)]
            self.atom_to_fg_membership = [[] for _ in range(self.n_atoms)]

    def _build_graph(self):  # Explanation: defines _build_graph, which represent reaction graphs and inject functional-group knowledge
        """Builds the graph attributes."""
        self.n_atoms = 0  # number of atoms  # Explanation: stores this value on the object for later model operations
        self.n_bonds = 0  # number of bonds  # Explanation: stores this value on the object for later model operations
        self.f_atoms = []  # mapping from atom index to atom features  # Explanation: stores this value on the object for later model operations
        self.f_atoms_raw = []
        # mapping from bond index to concat(in_atom, bond) features
        self.f_bonds = []  # Explanation: stores this value on the object for later model operations
        self.f_bond_attrs = []
        self.a2b = []  # mapping from atom index to incoming bond indices  # Explanation: stores this value on the object for later model operations
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from  # Explanation: stores this value on the object for later model operations
        self.b2revb = []  # mapping from bond index to the index of the reverse bond  # Explanation: stores this value on the object for later model operations

        self._empty_contextual_fg_metadata()
        self.f_fgs = []
        self.fg_names = []
        self.atoms = []  # Explanation: stores this value on the object for later model operations

        # Get atom features
        self.f_atoms_raw = [get_atom_features(  # Explanation: stores this value on the object for later model operations
            atom, rxn_class=self.rxn_class, use_rxn_class=self.use_rxn_class) for atom in self.mol.GetAtoms()]  # Explanation: assigns an intermediate value used by later computation
        self.f_atoms = [list(features) for features in self.f_atoms_raw]
        self.n_atoms = len(self.f_atoms)  # Explanation: stores this value on the object for later model operations
        for atom in self.mol.GetAtoms():  # Explanation: iterates over this collection to process each item
            self.atoms.append(atom.GetSymbol())  # Explanation: uses or updates this object state during computation
        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):  # Explanation: iterates over this collection to process each item
            self.a2b.append([])  # Explanation: uses or updates this object state during computation

        if self.fg_mode == "legacy":
            self.f_fgs, self.fg_names = match_fg(self.mol, self.use_rxn_class)  # Explanation: stores this value on the object for later model operations
        elif self.fg_mode == "contextual":
            self._build_contextual_fg_metadata()

        # add group knowledge
        if self.fg_mode == "legacy" and self.f_fgs:  # Explanation: checks this condition to choose the next execution path
            temp_tensor = torch.tensor(self.f_atoms)  # Explanation: assigns an intermediate value used by later computation
            f_fgs_tensor = torch.tensor(self.f_fgs)  # Explanation: assigns an intermediate value used by later computation
            fuse_f_atoms, self.attn_score = attention(temp_tensor, f_fgs_tensor)  # Explanation: assigns an intermediate value used by later computation
            self.f_atoms = fuse_f_atoms.tolist()  # Explanation: stores this value on the object for later model operations

        # Get bond features
        for a1 in range(self.n_atoms):  # Explanation: iterates over this collection to process each item
            for a2 in range(a1 + 1, self.n_atoms):  # Explanation: iterates over this collection to process each item
                bond = self.mol.GetBondBetweenAtoms(a1, a2)  # Explanation: assigns an intermediate value used by later computation

                if bond is None:  # Explanation: checks this condition to choose the next execution path
                    continue  # Explanation: skips the rest of this loop iteration

                f_bond = get_bond_features(bond)  # Explanation: assigns an intermediate value used by later computation

                self.f_bonds.append(self.f_atoms[a1] + f_bond)  # Explanation: uses or updates this object state during computation
                self.f_bonds.append(self.f_atoms[a2] + f_bond)  # Explanation: uses or updates this object state during computation
                self.f_bond_attrs.append(f_bond)
                self.f_bond_attrs.append(f_bond)

                # Update index mappings
                b1 = self.n_bonds  # Explanation: assigns an intermediate value used by later computation
                b2 = b1 + 1  # Explanation: assigns an intermediate value used by later computation
                self.a2b[a2].append(b1)  # b1 = a1 --> a2  # Explanation: stores this value on the object for later model operations
                self.b2a.append(a1)  # Explanation: uses or updates this object state during computation
                self.a2b[a1].append(b2)  # b2 = a2 --> a1  # Explanation: stores this value on the object for later model operations
                self.b2a.append(a2)  # Explanation: uses or updates this object state during computation
                self.b2revb.append(b2)  # Explanation: uses or updates this object state during computation
                self.b2revb.append(b1)  # Explanation: uses or updates this object state during computation
                self.n_bonds += 2  # Explanation: stores this value on the object for later model operations

class RxnGraph:  # Explanation: defines RxnGraph, reaction state containing a product graph and edit label
    """
    RxnGraph contains the information of a reaction, like reactants, products. The edits associated with the reaction are also captured in edit labels.
    """

    def __init__(
        self,
        prod_mol: Chem.Mol,
        edit_to_apply: Tuple,
        edit_atom: List = [],
        reac_mol: Chem.Mol = None,
        rxn_class: int = None,
        use_rxn_class: bool = False,
        fg_mode: str = "legacy",
        fg_context_radius: int = 1,
        fg_max_matches_per_pattern: int | None = None,
        fg_max_dist: int = 8,
    ) -> None:  # Explanation: computes an intermediate value for molecular graph editing
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        edit_to_apply: Tuple,
            Edits to apply to the product molecule
        edit_atom: List,
            Edit atom of product molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.prod_graph = MolGraph(  # Explanation: stores this value on the object for later model operations
            mol=prod_mol,
            rxn_class=rxn_class,
            use_rxn_class=use_rxn_class,
            fg_mode=fg_mode,
            fg_context_radius=fg_context_radius,
            fg_max_matches_per_pattern=fg_max_matches_per_pattern,
            fg_max_dist=fg_max_dist,
        )  # Explanation: assigns an intermediate value used by later computation
        if reac_mol is not None:  # Explanation: checks this condition to choose the next execution path
            self.reac_mol = reac_mol  # Explanation: stores this value on the object for later model operations
        self.edit_to_apply = edit_to_apply  # Explanation: stores this value on the object for later model operations
        self.edit_atom = edit_atom  # Explanation: stores this value on the object for later model operations
        self.rxn_class = rxn_class  # Explanation: stores this value on the object for later model operations

    def get_components(self, attrs: List = ['prod_graph', 'edit_to_apply', 'edit_atom']) -> Tuple:  # Explanation: defines get_components, which represent reaction graphs and inject functional-group knowledge
        """ 
        Returns the components associated with the reaction graph. 
        """
        attr_tuple = ()  # Explanation: assigns an intermediate value used by later computation
        for attr in attrs:  # Explanation: iterates over this collection to process each item
            if hasattr(self, attr):  # Explanation: checks this condition to choose the next execution path
                attr_tuple += (getattr(self, attr),)  # Explanation: assigns an intermediate value used by later computation
            else:  # Explanation: handles the fallback branch for the preceding condition
                print(f"Does not have attr {attr}")  # Explanation: prints progress or diagnostic information

        return attr_tuple  # Explanation: returns this computed result to the caller


class Vocab:  # Explanation: defines Vocab, maps edit tuples to integer ids
    """
    Vocab class to deal with vocabularies and other attributes.
    """

    def __init__(self, elem_list: List) -> None:  # Explanation: defines __init__, which represent reaction graphs and inject functional-group knowledge
        """
        Parameters
        ----------
        elem_list: List, default ATOM_LIST
            Element list used for setting up the vocab
        """
        self.elem_list = elem_list  # Explanation: stores this value on the object for later model operations
        if isinstance(elem_list, dict):  # Explanation: checks this condition to choose the next execution path
            self.elem_list = list(elem_list.keys())  # Explanation: stores this value on the object for later model operations
        self.elem_to_idx = {a: idx for idx, a in enumerate(self.elem_list)}  # Explanation: stores this value on the object for later model operations
        self.idx_to_elem = {idx: a for idx, a in enumerate(self.elem_list)}  # Explanation: stores this value on the object for later model operations

    def __getitem__(self, a_type: Tuple) -> int:  # Explanation: defines __getitem__, which represent reaction graphs and inject functional-group knowledge
        return self.elem_to_idx[a_type]  # Explanation: returns this computed result to the caller

    def get(self, elem: Tuple, idx: int = None) -> int:  # Explanation: defines get, which represent reaction graphs and inject functional-group knowledge
        """Returns the index of the element, else a None for missing element.

        Parameters
        ----------
        elem: str,
            Element to query
        idx: int, default None
            Index to return if element not in vocab
        """
        return self.elem_to_idx.get(elem, idx)  # Explanation: returns this computed result to the caller

    def get_elem(self, idx: int) -> Tuple:  # Explanation: defines get_elem, which represent reaction graphs and inject functional-group knowledge
        """Returns the element at given index.

        Parameters
        ----------
        idx: int,
            Index to return if element not in vocab
        """
        return self.idx_to_elem[idx]  # Explanation: returns this computed result to the caller

    def __len__(self) -> int:  # Explanation: defines __len__, which represent reaction graphs and inject functional-group knowledge
        return len(self.elem_list)  # Explanation: returns this computed result to the caller

    def get_index(self, elem: Tuple) -> int:  # Explanation: defines get_index, which represent reaction graphs and inject functional-group knowledge
        """Returns the index of the element.

        Parameters
        ----------
        elem: str,
            Element to query
        """
        return self.elem_to_idx[elem]  # Explanation: returns this computed result to the caller

    def size(self) -> int:  # Explanation: defines size, which represent reaction graphs and inject functional-group knowledge
        """Returns length of Vocab."""
        return len(self.elem_list)  # Explanation: returns this computed result to the caller
