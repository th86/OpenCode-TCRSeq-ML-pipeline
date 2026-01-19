#!/usr/bin/env python3
"""
Generate realistic TCR-seq example dataset with meaningful differences
between control and diseased samples.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path


def generate_cdr3_sequence(length_range=(8, 20)):
    """Generate realistic CDR3 amino acid sequences"""
    # Common CDR3 amino acids with typical frequencies
    amino_acids = [
        "C",
        "A",
        "S",
        "G",
        "T",
        "Y",
        "F",
        "L",
        "I",
        "V",
        "N",
        "D",
        "Q",
        "E",
        "K",
        "R",
        "H",
        "M",
        "P",
        "W",
    ]
    weights = [
        0.15,
        0.08,
        0.07,
        0.07,
        0.06,
        0.05,
        0.05,
        0.08,
        0.06,
        0.06,
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
        0.02,
        0.02,
        0.02,
        0.01,
    ]

    length = random.randint(*length_range)
    sequence = "".join(random.choices(amino_acids, weights=weights, k=length))

    # Ensure it starts with C and has a motif pattern
    if not sequence.startswith("C"):
        sequence = "C" + sequence[1:]

    # Add common motifs
    motifs = ["ASS", "SGG", "TDT", "YNEQ", "GEL", "QETQ", "VAG"]
    if len(sequence) > 8 and random.random() < 0.3:
        motif = random.choice(motifs)
        insert_pos = random.randint(2, len(sequence) - len(motif) - 2)
        sequence = sequence[:insert_pos] + motif + sequence[insert_pos + len(motif) :]

    return sequence


def generate_v_j_genes():
    """Generate realistic V and J gene combinations"""
    v_genes = [
        "TRBV12-3",
        "TRBV7-2",
        "TRBV20-1",
        "TRBV5-1",
        "TRBV6-5",
        "TRBV11-2",
        "TRBV28",
        "TRBV30",
        "TRBV4-1",
        "TRBV29-1",
        "TRBV24-1",
        "TRBV13-2",
        "TRBV9",
        "TRBV14",
        "TRBV27",
    ]

    j_genes = [
        "TRBJ2-7",
        "TRBJ1-2",
        "TRBJ2-1",
        "TRBJ1-5",
        "TRBJ2-3",
        "TRBJ1-3",
        "TRBJ2-5",
        "TRBJ1-1",
        "TRBJ2-2",
        "TRBJ1-6",
    ]

    return random.choice(v_genes), random.choice(j_genes)


def generate_control_sample(n_clones=1000, filename="control_rep1_01.csv"):
    """Generate control sample with polyclonal TCR repertoire"""
    data = []

    # Control samples: diverse, polyclonal repertoire
    clone_sizes = np.random.lognormal(mean=2, sigma=1.2, size=n_clones)
    clone_sizes = np.maximum(clone_sizes, 1).astype(int)

    # Generate some common clones
    for i in range(min(50, n_clones)):  # Top 50 clones are more frequent
        cdr3 = generate_cdr3_sequence(length_range=(12, 18))
        v_gene, j_gene = generate_v_j_genes()
        read_count = (
            random.randint(50, 500) * (50 - i) // 50 + 10
        )  # Decreasing frequency
        data.append([cdr3, v_gene, j_gene, read_count])

    # Generate many rare clones
    for i in range(50, n_clones):
        cdr3 = generate_cdr3_sequence(length_range=(8, 20))
        v_gene, j_gene = generate_v_j_genes()
        read_count = random.randint(1, 20)
        data.append([cdr3, v_gene, j_gene, read_count])

    df = pd.DataFrame(
        data, columns=["cdr3_amino_acid", "v_gene", "j_gene", "read_count"]
    )

    # Sort by read count (descending)
    df = df.sort_values("read_count", ascending=False).reset_index(drop=True)

    df.to_csv(filename, index=False)
    print(
        f"Generated {filename}: {len(df)} clones, {df['read_count'].sum()} total reads"
    )
    return df


def generate_diseased_sample(n_clones=800, filename="diseased_rep1_01.csv"):
    """Generate diseased sample with skewed/oligoclonal repertoire"""
    data = []

    # Diseased samples: oligoclonal expansion, skewed V/J usage
    n_expanded = random.randint(5, 15)  # Number of expanded clones

    # Generate expanded clones (very high frequency)
    for i in range(n_expanded):
        # Use more biased V genes for diseased
        v_bias = ["TRBV20-1", "TRBV5-1", "TRBV28"] if random.random() < 0.6 else None
        if v_bias:
            v_gene = random.choice(v_bias)
        else:
            v_gene, _ = generate_v_j_genes()
        _, j_gene = generate_v_j_genes()

        cdr3 = generate_cdr3_sequence(length_range=(14, 16))  # Slightly longer CDR3s
        read_count = random.randint(200, 2000)  # Much higher read counts
        data.append([cdr3, v_gene, j_gene, read_count])

    # Generate some medium frequency clones
    n_medium = min(100, n_clones - n_expanded)
    for i in range(n_medium):
        cdr3 = generate_cdr3_sequence(length_range=(10, 18))
        v_gene, j_gene = generate_v_j_genes()
        read_count = random.randint(20, 100)
        data.append([cdr3, v_gene, j_gene, read_count])

    # Generate remaining low frequency clones
    n_low = n_clones - n_expanded - n_medium
    for i in range(n_low):
        cdr3 = generate_cdr3_sequence(length_range=(8, 20))
        v_gene, j_gene = generate_v_j_genes()
        read_count = random.randint(1, 15)
        data.append([cdr3, v_gene, j_gene, read_count])

    df = pd.DataFrame(
        data, columns=["cdr3_amino_acid", "v_gene", "j_gene", "read_count"]
    )

    # Sort by read count (descending)
    df = df.sort_values("read_count", ascending=False).reset_index(drop=True)

    df.to_csv(filename, index=False)
    print(
        f"Generated {filename}: {len(df)} clones, {df['read_count'].sum()} total reads"
    )
    return df


def calculate_sample_stats(df, sample_name):
    """Calculate basic statistics for the sample"""
    total_reads = df["read_count"].sum()
    df["frequency"] = df["read_count"] / total_reads

    # Shannon diversity
    frequencies = df["frequency"].values
    shannon = -np.sum(frequencies * np.log(frequencies + 1e-10))

    # Clonality (1 - normalized Shannon)
    max_shannon = np.log(len(df))
    clonality = 1 - (shannon / max_shannon) if max_shannon > 0 else 0

    # Top clone frequency
    top_freq = df["frequency"].iloc[0]

    print(f"{sample_name} Stats:")
    print(f"  Total reads: {total_reads:,}")
    print(f"  Unique clones: {len(df):,}")
    print(f"  Shannon diversity: {shannon:.3f}")
    print(f"  Clonality: {clonality:.3f}")
    print(f"  Top clone frequency: {top_freq:.3f}")
    print(f"  Top 10 clones: {df['frequency'].head(10).sum():.3f}")
    print()


def main():
    """Generate comprehensive example dataset"""
    np.random.seed(42)
    random.seed(42)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Generating realistic TCR-seq dataset...")
    print("=" * 50)

    # Generate multiple replicates for robust analysis
    samples = [
        # Control samples - polyclonal, diverse
        ("control_rep1_01.csv", 1200, "polyclonal"),
        ("control_rep2_01.csv", 1100, "polyclonal"),
        ("control_rep3_01.csv", 1300, "polyclonal"),
        # Diseased samples - oligoclonal, skewed
        ("diseased_rep1_01.csv", 800, "oligoclonal"),
        ("diseased_rep2_01.csv", 900, "oligoclonal"),
        ("diseased_rep3_01.csv", 750, "oligoclonal"),
    ]

    generated_samples = {}

    for filename, n_clones, sample_type in samples:
        if "control" in filename:
            df = generate_control_sample(n_clones, filename)
            sample_name = filename.replace(".csv", "")
        else:
            df = generate_diseased_sample(n_clones, filename)
            sample_name = filename.replace(".csv", "")

        generated_samples[sample_name] = df
        calculate_sample_stats(df, sample_name)

    # Calculate group-level statistics
    print("Group-level Analysis:")
    print("=" * 30)

    control_samples = [v for k, v in generated_samples.items() if "control" in k]
    diseased_samples = [v for k, v in generated_samples.items() if "diseased" in k]

    # Average diversity metrics
    control_diversities = []
    control_clonalities = []
    for df in control_samples:
        total_reads = df["read_count"].sum()
        df["frequency"] = df["read_count"] / total_reads
        frequencies = df["frequency"].values
        shannon = -np.sum(frequencies * np.log(frequencies + 1e-10))
        max_shannon = np.log(len(df))
        clonality = 1 - (shannon / max_shannon) if max_shannon > 0 else 0
        control_diversities.append(shannon)
        control_clonalities.append(clonality)

    diseased_diversities = []
    diseased_clonalities = []
    for df in diseased_samples:
        total_reads = df["read_count"].sum()
        df["frequency"] = df["read_count"] / total_reads
        frequencies = df["frequency"].values
        shannon = -np.sum(frequencies * np.log(frequencies + 1e-10))
        max_shannon = np.log(len(df))
        clonality = 1 - (shannon / max_shannon) if max_shannon > 0 else 0
        diseased_diversities.append(shannon)
        diseased_clonalities.append(clonality)

    print(f"Control Group:")
    print(
        f"  Mean Shannon diversity: {np.mean(control_diversities):.3f} ± {np.std(control_diversities):.3f}"
    )
    print(
        f"  Mean clonality: {np.mean(control_clonalities):.3f} ± {np.std(control_clonalities):.3f}"
    )

    print(f"Diseased Group:")
    print(
        f"  Mean Shannon diversity: {np.mean(diseased_diversities):.3f} ± {np.std(diseased_diversities):.3f}"
    )
    print(
        f"  Mean clonality: {np.mean(diseased_clonalities):.3f} ± {np.std(diseased_clonalities):.3f}"
    )

    # Effect size
    diversity_diff = np.mean(control_diversities) - np.mean(diseased_diversities)
    clonality_diff = np.mean(diseased_clonalities) - np.mean(control_clonalities)

    print(f"\nExpected Differences:")
    print(f"  Shannon diversity difference: {diversity_diff:.3f}")
    print(f"  Clonality difference: {clonality_diff:.3f}")

    print(f"\nDataset generated successfully! Files saved to data/")
    print(f"Run the workflow with: python main.py --data_dir ./data")


if __name__ == "__main__":
    main()
