"""Build Protenix template features from designed coordinates.

This module injects designed structure coordinates as template features
for Protenix scoring, bypassing the normal template search pipeline.
"""

import numpy as np
import torch

from protenix.data.constants import (
    ATOM14,
    ATOM14_PADDED,
    ATOM37,
    ATOM37_ORDER,
    PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37,
    PROTEIN_TYPES_ONE_LETTER,
    PROTEIN_COMMON_ONE_TO_THREE,
    RESTYPE_PSEUDOBETA_INDEX,
    RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX,
)
from protenix.data.template.template_featurizer import Templates
from protenix.data.template.template_utils import (
    DistogramFeaturesConfig,
    TemplateFeatures,
)


def build_template_features_from_structure(
    coords_atom14: np.ndarray,
    res_types: np.ndarray,
    num_templates: int = 4,
) -> dict:
    """Build Protenix template feature dict from designed coordinates.

    Takes atom14 coordinates from a designed structure and produces the
    full set of template features expected by the Protenix model.

    Processing steps:
    1. atom14 -> atom37 (via residue-type-specific mapping)
    2. atom37 -> dense 24-atom (via PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37)
    3. Build Templates dataclass
    4. Call Templates.as_protenix_dict() for distogram, unit_vector, etc.
    5. Rename keys to model expectations
    6. Pad first template with real data, rest with zeros
    7. Convert to tensors

    Args:
        coords_atom14: [N_token, 14, 3] atom14 coordinates.
        res_types: [N_token] int residue type indices (Protenix convention:
                   0=ALA, 1=ARG, ..., 19=VAL, 20=UNK).
        num_templates: number of template slots (Protenix expects 4).

    Returns:
        Dict with keys matching make_dummy_feature():
            template_restype: [num_templates, N_token]
            template_all_atom_mask: [num_templates, N_token, 37]
            template_all_atom_positions: [num_templates, N_token, 37, 3]
            template_pseudo_beta_mask: [num_templates, N_token, N_token]
            template_backbone_frame_mask: [num_templates, N_token, N_token]
            template_distogram: [num_templates, N_token, N_token, 39]
            template_unit_vector: [num_templates, N_token, N_token, 3]
    """
    N = coords_atom14.shape[0]

    # Step 1: atom14 -> atom37
    coords37 = np.zeros((N, 37, 3), dtype=np.float32)
    mask37 = np.zeros((N, 37), dtype=np.float32)

    for i in range(N):
        rt = int(res_types[i])
        if rt >= len(PROTEIN_TYPES_ONE_LETTER):
            continue
        one_letter = PROTEIN_TYPES_ONE_LETTER[rt]
        resname = PROTEIN_COMMON_ONE_TO_THREE[one_letter]
        atom_names = ATOM14.get(resname, ())
        for j, aname in enumerate(atom_names):
            if aname in ATOM37_ORDER:
                a37_idx = ATOM37_ORDER[aname]
                coords37[i, a37_idx] = coords_atom14[i, j]
                # Mark as present if coords are non-zero
                if np.any(coords_atom14[i, j] != 0):
                    mask37[i, a37_idx] = 1.0

    # Step 2: atom37 -> dense24 (using same mapping as fix_template_features)
    dense_atom_indices = np.take(
        PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37,
        res_types.astype(np.int32),
        axis=0,
    )  # [N, 24]

    dense_mask = np.take_along_axis(
        mask37, dense_atom_indices, axis=1
    )  # [N, 24]
    dense_coords = np.take_along_axis(
        coords37, dense_atom_indices[..., None], axis=1
    )  # [N, 24, 3]
    dense_coords *= dense_mask[..., None]

    # Step 3: Build Templates dataclass (single template)
    aatype = res_types.astype(np.int32)[None, ...]  # [1, N]
    atom_positions = dense_coords[None, ...].astype(np.float32)  # [1, N, 24, 3]
    atom_mask = dense_mask[None, ...].astype(np.int32)  # [1, N, 24]

    templates = Templates(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
    )

    # Step 4: Compute derived features (distogram, unit_vector, etc.)
    protenix_features = templates.as_protenix_dict()

    # Step 5: Build output with correct key names and padding
    # Model expects atom37 format for positions/mask at the feature dict level
    # Plus the derived 2D features from as_protenix_dict()

    # Pad to num_templates (first slot has real data, rest are zeros)
    template_restype = _pad_to_templates(aatype, num_templates, fill_value=31)
    template_all_atom_mask = _pad_to_templates(
        mask37[None, ...].astype(np.int32), num_templates, fill_value=0
    )
    template_all_atom_positions = _pad_to_templates(
        coords37[None, ...].astype(np.float32), num_templates, fill_value=0.0
    )
    template_pseudo_beta_mask = _pad_to_templates(
        protenix_features["template_pseudo_beta_mask"], num_templates, fill_value=0.0
    )
    template_backbone_frame_mask = _pad_to_templates(
        protenix_features["template_backbone_frame_mask"], num_templates, fill_value=0.0
    )
    template_distogram = _pad_to_templates(
        protenix_features["template_distogram"], num_templates, fill_value=0.0
    )
    template_unit_vector = _pad_to_templates(
        protenix_features["template_unit_vector"], num_templates, fill_value=0.0
    )

    # Step 7: Convert to tensors
    return {
        "template_restype": torch.from_numpy(template_restype).long(),
        "template_all_atom_mask": torch.from_numpy(template_all_atom_mask).long(),
        "template_all_atom_positions": torch.from_numpy(template_all_atom_positions).float(),
        "template_pseudo_beta_mask": torch.from_numpy(template_pseudo_beta_mask).float(),
        "template_backbone_frame_mask": torch.from_numpy(template_backbone_frame_mask).float(),
        "template_distogram": torch.from_numpy(template_distogram).float(),
        "template_unit_vector": torch.from_numpy(template_unit_vector).float(),
    }


def _pad_to_templates(
    arr: np.ndarray,
    num_templates: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Pad a [1, ...] array to [num_templates, ...] with fill_value.

    First template slot gets the real data, remaining slots are filled.
    """
    if arr.shape[0] >= num_templates:
        return arr[:num_templates]

    pad_shape = (num_templates - arr.shape[0],) + arr.shape[1:]
    padding = np.full(pad_shape, fill_value, dtype=arr.dtype)
    return np.concatenate([arr, padding], axis=0)
