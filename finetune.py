import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    import fast_hdbscan
    FAST_HDBSCAN_AVAILABLE = True
    print("[+] fast_hdbscan available, using multicore optimized HDBSCAN")
except ImportError:
    FAST_HDBSCAN_AVAILABLE = False
    print("[-] fast_hdbscan not available, please install fast-hdbscan package")
    print("    pip install fast-hdbscan")
    exit(1)

from pretrain import MalwareDataset, extract_statistical_features, load_dataset
from torch.utils.data import DataLoader

def load_features_from_pickle(pickle_path):

    print(f"[*] Loading features from {pickle_path}...")

    df = pd.read_pickle(pickle_path)

    files = df['filename'].tolist()
    labels = df['label'].values

    feature_columns = [col for col in df.columns if col.startswith('feature_')]
    features = df[feature_columns].values

    print(f"[+] Successfully loaded features for {len(files)} samples")
    print(f"    Feature dimension: {features.shape[1]}")

    return features, labels, files

def extract_features_with_labels(data_dir, metadata_file, max_file_size=256*1024):

    X, y, files = load_dataset(data_dir, metadata_file, max_file_size)
    return X, y, files

def filter_malicious_samples(features, labels, files):

    print("[*] Filtering malware samples for family clustering...")

    malicious_indices = np.where(labels == 1)[0]

    malicious_features = features[malicious_indices]
    malicious_labels = labels[malicious_indices]
    malicious_files = [files[i] for i in malicious_indices]

    print(f"[+] Filtering completed:")
    print(f"    Malware samples: {len(malicious_files)}")
    print(f"    Benign samples: {len(files) - len(malicious_files)}")

    return malicious_features, malicious_labels, malicious_files

def perform_hdbscan_clustering(features, min_cluster_size=50, min_samples=10):

    print("[*] Performing clustering analysis using HDBSCAN...")
    print(f"    [*] Feature dimension: {features.shape[1]}")

    print("    [*] Using fast_hdbscan multicore optimized version")
    clusterer = fast_hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(features)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    print(f"[+] Clustering completed:")
    print(f"    Total clusters: {n_clusters}")
    print(f"    Noise points: {np.sum(labels == -1) if -1 in labels else 0}")
    print(f"    Total samples: {len(labels)}")

    return labels, clusterer

def analyze_clusters(files, labels, min_family_size=20, treat_noise_as_family=False):

    print("[*] Analyzing clustering results to discover new families...")

    unique_labels, counts = np.unique(labels, return_counts=True)

    family_clusters = {}
    noise_count = 0
    small_cluster_count = 0
    noise_families = 0

    for label, count in zip(unique_labels, counts):
        if label == -1:
            noise_count = count
            if treat_noise_as_family and count >= min_family_size:
                family_clusters[int(label)] = int(count)
                noise_families += 1
        elif count >= min_family_size:
            family_clusters[int(label)] = int(count)
        else:
            small_cluster_count += 1
            if treat_noise_as_family:
                family_clusters[int(label)] = int(count)
                noise_families += 1 if label == -1 else 0

    noise_as_family_text = f" (of which {noise_families} are noise point families)" if treat_noise_as_family else ""
    print(f"[+] Family analysis completed:")
    print(f"    Identified {len(family_clusters)} potential malware families{noise_as_family_text}")
    print(f"    Noise samples: {noise_count}")
    print(f"    Small clusters (less than {min_family_size} samples): {small_cluster_count}")

    return {
        'families': family_clusters,
        'noise_count': noise_count,
        'small_clusters': small_cluster_count
    }

def visualize_clusters(features, labels, save_path, plot_pca=False):

    print("[*] Generating clustering visualization...")
    print(f"    [*] Feature dimension: {features.shape[1]}")

    original_features = features.copy()

    if features.shape[1] > 50:
        print("    [*] High feature dimension, reducing to 50 dimensions using PCA")
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(features)

    if features.shape[0] > 10000:
        print("    [*] Large number of samples, randomly sampling 10000 points for visualization")
        indices = np.random.choice(features.shape[0], 10000, replace=False)
        features = features[indices]
        original_features = original_features[indices]
        labels = labels[indices]

    print("    [*] Using t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    colors = ['gray' if label == -1 else None for label in labels]
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Malware Clustering Results Visualization (HDBSCAN + t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[+] Clustering visualization saved to: {save_path}")

    if plot_pca:
        pca_save_path = save_path.replace('.png', '_pca.png')
        print("    [*] Generating additional PCA dimensionality reduction visualization...")
        pca_2d = PCA(n_components=2, random_state=42)
        features_pca_2d = pca_2d.fit_transform(original_features)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_pca_2d[:, 0], features_pca_2d[:, 1], c=labels, cmap='tab20', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Malware Clustering Results Visualization (HDBSCAN + PCA)')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.grid(True)
        plt.savefig(pca_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[+] PCA visualization saved to: {pca_save_path}")

def explain_clustering_discrepancy(features, labels, sample_indices=None, num_samples=5):

    print("[*] Explaining visualization discrepancies in clustering results...")

    if sample_indices is None:
        non_noise_indices = np.where(labels != -1)[0]
        if len(non_noise_indices) > 0:
            sample_indices = np.random.choice(non_noise_indices,
                                            min(num_samples, len(non_noise_indices)),
                                            replace=False)
        else:
            print("    [!] No samples found in non-noise clusters")
            return

    print(f"    [*] Analyzing {len(sample_indices)} samples in different dimensional spaces")

    print("    [*] Euclidean distances between samples in high-dimensional space:")
    for i, idx in enumerate(sample_indices):
        same_cluster = np.where(labels == labels[idx])[0]
        if len(same_cluster) > 1:
            same_cluster_distances = [np.linalg.norm(features[idx] - features[j])
                                   for j in same_cluster if j != idx]
            avg_same_cluster_dist = np.mean(same_cluster_distances)
            print(f"        Sample {idx} average distance to same cluster samples: {avg_same_cluster_dist:.4f}")

        diff_cluster = np.where(labels != labels[idx])[0]
        diff_cluster = diff_cluster[diff_cluster != idx]
        if len(diff_cluster) > 0:
            diff_cluster_distances = [np.linalg.norm(features[idx] - features[j])
                                   for j in diff_cluster[:100]]
            avg_diff_cluster_dist = np.mean(diff_cluster_distances)
            print(f"        Sample {idx} average distance to different cluster samples: {avg_diff_cluster_dist:.4f}")

    print("    [*] Notes:")
    print("        1. t-SNE is a nonlinear dimensionality reduction method that prioritizes preserving local structure over global distances")
    print("        2. Points that are far apart in high-dimensional space may appear close in t-SNE")
    print("        3. Points that are close in high-dimensional space may be mapped far apart in t-SNE")
    print("        4. HDBSCAN performs clustering based on density in high-dimensional space, not distances in reduced space")

def save_clustering_results(files, labels, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    file_cluster_mapping = {}
    for file, label in zip(files, labels):
        file_cluster_mapping[file] = int(label)

    mapping_path = os.path.join(save_dir, 'file_cluster_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(file_cluster_mapping, f, indent=2, ensure_ascii=False)

    print(f"[+] File-cluster mapping saved to: {mapping_path}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_stats = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    stats_path = os.path.join(save_dir, 'cluster_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_stats, f, indent=2, ensure_ascii=False)

    print(f"[+] Cluster statistics saved to: {stats_path}")

def identify_new_families(files, labels, save_dir, min_family_size=20, treat_noise_as_family=False):

    print("[*] Identifying newly discovered malware families...")

    cluster_analysis = analyze_clusters(files, labels, min_family_size, treat_noise_as_family)

    family_names = {}
    for cluster_id, count in cluster_analysis['families'].items():
        family_names[cluster_id] = f"Malware_Family_{cluster_id}_Size{count}"

    print(f"[+] Identified {len(family_names)} new families")

    family_mapping_path = os.path.join(save_dir, 'family_names_mapping.json')
    with open(family_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(family_names, f, indent=2, ensure_ascii=False)

    print(f"[+] Family name mapping saved to: {family_mapping_path}")

    return family_names

def main(args):

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    print("\n[*] Step 1/3: Preparing dataset...")

    features_pkl_path = args.features_path if args.features_path else os.path.join('.', 'extracted_features.pkl')

    if not os.path.exists(features_pkl_path):
        print(f"[!] Error: Feature file not found {features_pkl_path}")
        print("    Please run pretrain.py first to generate feature file")
        metadata_file = os.path.join(args.data_dir, 'metadata.json')
        if os.path.exists(args.data_dir) and os.path.exists(metadata_file):
            print(f"    [*] Attempting to generate features from data directory: {args.data_dir}")
            features, labels, files = extract_features_with_labels(
                args.data_dir, metadata_file, args.max_file_size
            )
            feature_df = pd.DataFrame(features)
            feature_df.columns = [f'feature_{i}' for i in range(features.shape[1])]
            feature_df['label'] = labels
            feature_df['filename'] = files
            feature_df.to_pickle(features_pkl_path)
            print(f"    [+] Features saved to {features_pkl_path}")
        else:
            return
    else:
        print("\n[*] Step 2/3: Loading features...")
        features, labels, files = load_features_from_pickle(features_pkl_path)

    if features.size == 0:
        print("[!] Failed to load any features, cannot perform clustering. Please check feature file.")
        return

    print("\n[*] Step 3/5: Filtering malware samples...")
    malicious_features, malicious_labels, malicious_files = filter_malicious_samples(features, labels, files)

    if malicious_features.size == 0:
        print("[!] No malware samples in dataset, cannot perform family clustering analysis.")
        return

    print("\n[*] Step 4/5: Performing HDBSCAN clustering analysis...")
    cluster_labels, clusterer = perform_hdbscan_clustering(
        malicious_features,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )

    print("\n[*] Step 5/6: Analyzing clusters and identifying new families...")
    family_analysis = analyze_clusters(malicious_files, cluster_labels, args.min_family_size, args.treat_noise_as_family)
    family_names = identify_new_families(malicious_files, cluster_labels, args.save_dir, args.min_family_size, args.treat_noise_as_family)

    print("\n[*] Step 6/6: Saving clustering results...")
    save_clustering_results(malicious_files, cluster_labels, args.save_dir)

    visualize_clusters(malicious_features, cluster_labels,
                      os.path.join(args.save_dir, 'hdbscan_clustering_visualization.png'),
                      plot_pca=args.plot_pca)

    if args.explain_discrepancy:
        explain_clustering_discrepancy(malicious_features, cluster_labels)

    try:
        if len(np.unique(cluster_labels)) > 1 and -1 not in cluster_labels:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            sil_score = silhouette_score(malicious_features, cluster_labels)
            ch_score = calinski_harabasz_score(malicious_features, cluster_labels)
            print(f"\n[*] Clustering quality assessment:")
            print(f"    Silhouette Score: {sil_score:.4f}")
            print(f"    Calinski-Harabasz Index: {ch_score:.4f}")
    except Exception as e:
        print(f"[!] Unable to calculate clustering quality metrics: {e}")

    print("\n[+] HDBSCAN clustering fine-tuning completed!")
    print(f"[*] Processed {len(malicious_files)} malware files")
    print(f"[*] Discovered {len(family_analysis['families'])} new families")
    print(f"[*] Identified {family_analysis['noise_count']} noise points")
    print(f"[*] Results saved to: {args.save_dir}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="HDBSCAN-based malware family discovery script")

    parser.add_argument('--data-dir', type=str, default='./data/processed_lightgbm',
                       help='Data directory (default: ./data/processed_lightgbm)')
    parser.add_argument('--features-path', type=str, default='',
                       help='Feature pickle file path (default: ./extracted_features.pkl)')
    parser.add_argument('--save-dir', type=str, default='./hdbscan_cluster_results',
                       help='Directory to save clustering results')

    parser.add_argument('--max-file-size', type=int, default=256 * 1024,
                       help='Maximum input file size in bytes')
    parser.add_argument('--min-cluster-size', type=int, default=50,
                       help='HDBSCAN minimum cluster size (default: 50)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='HDBSCAN minimum samples for core points (default: 10)')
    parser.add_argument('--min-family-size', type=int, default=20,
                       help='Minimum samples to define a family (default: 20)')
    parser.add_argument('--plot-pca', action='store_true',
                       help='Generate additional PCA dimensionality reduction visualization')
    parser.add_argument('--explain-discrepancy', action='store_true',
                       help='Explain why nearby points belong to different clusters')
    parser.add_argument('--treat-noise-as-family', action='store_true',
                       help='Treat noise points as separate families (must meet minimum family size requirement)')

    args = parser.parse_args()
    main(args)
