"""Standalone scoring script for designed antibody structures using Protenix.

Reads designed structures + sequences, runs Protenix forward pass,
and extracts confidence metrics (iptm, ptm, plddt, ranking_score).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def score_designs(
    input_dir: str,
    output_dir: str,
    model_name: str = "protenix_base_default_v1.0.0",
    use_template: bool = True,
    max_crop_length: int = 768,
):
    """Score designed structures using Protenix.

    For each design in input_dir:
    1. Build Protenix input JSON from CIF structure
    2. Run through InferenceRunner featurization pipeline
    3. If use_template: replace dummy template features with designed coords
    4. Forward pass
    5. Extract confidence metrics
    6. Write confidence.json + prediction CIF

    Args:
        input_dir: directory with designed structures (CIF) and sequences (FASTA).
        output_dir: directory for scoring outputs.
        model_name: Protenix model checkpoint name.
        use_template: whether to inject designed coords as template features.
        max_crop_length: maximum token length for cropping.
    """
    from runner.batch_inference import get_default_runner
    from protenix.data.inference import cif_to_input_json
    from protenix.data.inference.infer_dataloader import get_inference_dataloader

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize runner
    runner = get_default_runner(
        model_name=model_name,
        use_template=use_template,
        n_sample=1,
        n_step=200,
        n_cycle=10,
    )

    # Process each spec directory
    for spec_dir in sorted(input_dir.iterdir()):
        if not spec_dir.is_dir():
            continue

        spec_id = spec_dir.name
        spec_output = output_dir / spec_id
        spec_output.mkdir(parents=True, exist_ok=True)

        # Find FASTA files (each represents a designed sample)
        fasta_files = sorted(spec_dir.glob("*.fasta"))

        for fasta_path in fasta_files:
            sample_id = fasta_path.stem
            sample_output = spec_output / sample_id
            sample_output.mkdir(parents=True, exist_ok=True)

            # Find the corresponding CIF from step2
            cif_path = _find_structure_cif(input_dir, spec_id, sample_id)
            if cif_path is None:
                logger.warning(f"No CIF found for {spec_id}/{sample_id}, skipping")
                continue

            # Read designed sequence from FASTA
            sequences = _read_fasta(str(fasta_path))
            if not sequences:
                logger.warning(f"No sequences in {fasta_path}, skipping")
                continue

            # Score the best beam (first sequence)
            best_seq = sequences[0][1]

            try:
                confidence = _score_single_design(
                    runner=runner,
                    cif_path=str(cif_path),
                    sequence=best_seq,
                    sample_name=f"{spec_id}_{sample_id}",
                    use_template=use_template,
                    output_dir=str(sample_output),
                )

                # Write confidence
                conf_path = sample_output / "confidence.json"
                with open(conf_path, "w") as f:
                    json.dump(confidence, f, indent=2)

                logger.info(
                    f"Scored {spec_id}/{sample_id}: "
                    f"iptm={confidence.get('iptm', 'N/A'):.4f}, "
                    f"ptm={confidence.get('ptm', 'N/A'):.4f}, "
                    f"ranking_score={confidence.get('ranking_score', 'N/A'):.4f}"
                )

            except Exception as e:
                logger.error(f"Failed to score {spec_id}/{sample_id}: {e}")
                # Write error file
                with open(sample_output / "error.txt", "w") as f:
                    f.write(str(e))


def _score_single_design(
    runner,
    cif_path: str,
    sequence: str,
    sample_name: str,
    use_template: bool,
    output_dir: str,
) -> dict:
    """Score a single design through Protenix.

    Returns dict with confidence metrics.
    """
    from protenix.data.inference import cif_to_input_json
    from protenix.data.inference.infer_dataloader import get_inference_dataloader

    # Build input JSON from CIF
    input_json = cif_to_input_json(cif_path, sample_name=sample_name)

    # Write temporary JSON for the dataloader
    tmp_json_path = os.path.join(output_dir, "input.json")
    with open(tmp_json_path, "w") as f:
        json.dump([input_json], f)

    # Create dataloader
    dataloader = get_inference_dataloader(
        configs=runner.configs,
        json_path=tmp_json_path,
    )

    # Run inference
    for batch in dataloader:
        data, atom_array, error_msg = batch[0]
        if error_msg:
            raise RuntimeError(f"Data processing error: {error_msg}")

        # Optionally inject template features from designed coords
        if use_template:
            _inject_template_if_available(data, cif_path)

        prediction = runner.predict(data)

        # Extract confidence metrics
        confidence = _extract_confidence(prediction)

        # Clean up temp file
        os.remove(tmp_json_path)

        return confidence

    raise RuntimeError("No data produced by dataloader")


def _inject_template_if_available(data: dict, cif_path: str):
    """Replace dummy template features with actual designed coordinates."""
    try:
        from protenix.data.template.inject_template import (
            build_template_features_from_structure,
        )
        from abpipeline.io_utils import read_chains_from_cif

        # Try to load atom14 coords from NPZ if available
        npz_path = Path(cif_path).with_suffix(".npz")
        if npz_path.exists():
            npz_data = np.load(str(npz_path), allow_pickle=True)
            if "coords" in npz_data and "res_type" in npz_data:
                coords_atom14 = npz_data["coords"]
                res_types = npz_data["res_type"]
                if res_types.ndim == 2:
                    res_types = res_types.argmax(axis=-1)

                template_features = build_template_features_from_structure(
                    coords_atom14=coords_atom14,
                    res_types=res_types,
                )

                # Replace template features in the input data
                feat_dict = data["input_feature_dict"]
                for key, value in template_features.items():
                    if key in feat_dict:
                        feat_dict[key] = value
                return

        logger.debug(f"No NPZ found for template injection at {npz_path}")

    except ImportError:
        logger.debug("Template injection not available (missing imports)")
    except Exception as e:
        logger.warning(f"Template injection failed: {e}")


def _extract_confidence(prediction: dict) -> dict:
    """Extract summary confidence metrics from Protenix prediction."""
    confidence = {}

    if "summary_confidence" in prediction:
        sc = prediction["summary_confidence"]
        # summary_confidence is a list of per-sample dicts
        if isinstance(sc, list) and len(sc) > 0:
            sample_conf = sc[0]
            for key in ["iptm", "ptm", "plddt", "ranking_score", "gpde"]:
                if key in sample_conf:
                    val = sample_conf[key]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    confidence[key] = float(val)

    return confidence


def _find_structure_cif(
    input_dir: Path,
    spec_id: str,
    sample_id: str,
) -> Path | None:
    """Find the structure CIF file for a given spec and sample."""
    # Look in step2 output first
    step2_dir = input_dir.parent / "step2_boltzgen" / spec_id / "uncropped"
    cif_path = step2_dir / f"{sample_id}.cif"
    if cif_path.exists():
        return cif_path

    # Look in the input directory itself
    cif_path = input_dir / spec_id / f"{sample_id}.cif"
    if cif_path.exists():
        return cif_path

    # Search more broadly
    for p in (input_dir / spec_id).rglob(f"*{sample_id}*.cif"):
        return p

    return None


def _read_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """Read sequences from a FASTA file.

    Returns list of (header, sequence) tuples.
    """
    sequences = []
    header = None
    seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

    if header is not None:
        sequences.append((header, "".join(seq_lines)))

    return sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Score designs with Protenix")
    parser.add_argument("--input_dir", required=True, help="Input directory with designs")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--model_name",
        default="protenix_base_default_v1.0.0",
        help="Model checkpoint name",
    )
    parser.add_argument("--use_template", action="store_true", help="Inject template features")
    parser.add_argument("--max_crop_length", type=int, default=768, help="Max crop length")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    score_designs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        use_template=args.use_template,
        max_crop_length=args.max_crop_length,
    )
