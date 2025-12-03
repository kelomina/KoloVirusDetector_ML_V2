import os
import json
import numpy as np
import lightgbm as lgb
from pathlib import Path
import csv
import hashlib
import sys
import gzip
import tempfile
import shutil
import argparse

from feature_extractor_enhanced import extract_features_in_memory

BASE_DIR = getattr(sys, '_MEIPASS', os.path.abspath('.'))

def extract_statistical_features(byte_sequence, pe_features):
    byte_array = np.array(byte_sequence, dtype=np.uint8)
    features = []

    features.extend([
        float(np.mean(byte_array)),
        float(np.std(byte_array)),
        float(np.min(byte_array)),
        float(np.max(byte_array)),
        float(np.median(byte_array)),
        float(np.percentile(byte_array, 25)),
        float(np.percentile(byte_array, 75)),
    ])

    features.extend([
        int(np.sum(byte_array == 0)),
        int(np.sum(byte_array == 0xFF)),
        int(np.sum(byte_array == 0x90)),
        int(np.sum((byte_array >= 32) & (byte_array <= 126))),
    ])

    hist, _ = np.histogram(byte_array, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    features.append(float(entropy))

    length = len(byte_array)
    if length >= 3:
        one_third = length // 3
        segments = [
            byte_array[:one_third],
            byte_array[one_third:2 * one_third],
            byte_array[2 * one_third:],
        ]
    else:
        segments = [byte_array, byte_array, byte_array]

    for seg in segments:
        if len(seg) == 0:
            seg_mean = 0.0
            seg_std = 0.0
            seg_entropy = 0.0
        else:
            seg_mean = float(np.mean(seg))
            seg_std = float(np.std(seg))
            seg_hist, _ = np.histogram(seg, bins=256, range=(0, 255), density=True)
            seg_hist = seg_hist[seg_hist > 0]
            seg_entropy = -np.sum(seg_hist * np.log2(seg_hist)) if len(seg_hist) > 0 else 0.0
        features.extend([seg_mean, seg_std, seg_entropy])

    chunk_size = max(1, len(byte_array) // 10)
    chunk_means = []
    chunk_stds = []
    for i in range(10):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 9 else len(byte_array)
        chunk = byte_array[start_idx:endidx]
        if len(chunk) > 0:
            chunk_means.append(float(np.mean(chunk)))
            chunk_stds.append(float(np.std(chunk)))
        else:
            chunk_means.append(0.0)
            chunk_stds.append(0.0)

    features.extend(chunk_means)
    features.extend(chunk_stds)

    chunk_means = np.array(chunk_means, dtype=np.float32)
    chunk_stds = np.array(chunk_stds, dtype=np.float32)

    if len(chunk_means) > 1:
        mean_diffs = np.diff(chunk_means)
        std_diffs = np.diff(chunk_stds)

        features.extend([
            float(np.mean(np.abs(mean_diffs))),
            float(np.std(mean_diffs)),
            float(np.max(mean_diffs)),
            float(np.min(mean_diffs)),
        ])
        features.extend([
            float(np.mean(np.abs(std_diffs))),
            float(np.std(std_diffs)),
            float(np.max(std_diffs)),
            float(np.min(std_diffs)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
        features.extend([0.0, 0.0, 0.0, 0.0])

    features.extend(pe_features.tolist())

    return np.array(features, dtype=np.float32)

BASE_DIR = getattr(sys, '_MEIPASS', os.path.abspath('.'))

def extract_statistical_features(byte_sequence, pe_features):
    byte_array = np.array(byte_sequence, dtype=np.uint8)
    features = []

    features.extend([
        float(np.mean(byte_array)),
        float(np.std(byte_array)),
        float(np.min(byte_array)),
        float(np.max(byte_array)),
        float(np.median(byte_array)),
        float(np.percentile(byte_array, 25)),
        float(np.percentile(byte_array, 75)),
    ])

    features.extend([
        int(np.sum(byte_array == 0)),
        int(np.sum(byte_array == 0xFF)),
        int(np.sum(byte_array == 0x90)),
        int(np.sum((byte_array >= 32) & (byte_array <= 126))),
    ])

    hist, _ = np.histogram(byte_array, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    features.append(float(entropy))

    length = len(byte_array)
    if length >= 3:
        one_third = length // 3
        segments = [
            byte_array[:one_third],
            byte_array[one_third:2 * one_third],
            byte_array[2 * one_third:],
        ]
    else:
        segments = [byte_array, byte_array, byte_array]

    for seg in segments:
        if len(seg) == 0:
            seg_mean = 0.0
            seg_std = 0.0
            seg_entropy = 0.0
        else:
            seg_mean = float(np.mean(seg))
            seg_std = float(np.std(seg))
            seg_hist, _ = np.histogram(seg, bins=256, range=(0, 255), density=True)
            seg_hist = seg_hist[seg_hist > 0]
            seg_entropy = -np.sum(seg_hist * np.log2(seg_hist)) if len(seg_hist) > 0 else 0.0
        features.extend([seg_mean, seg_std, seg_entropy])

    chunk_size = max(1, len(byte_array) // 10)
    chunk_means = []
    chunk_stds = []
    for i in range(10):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 9 else len(byte_array)
        chunk = byte_array[start_idx:end_idx]
        if len(chunk) > 0:
            chunk_means.append(float(np.mean(chunk)))
            chunk_stds.append(float(np.std(chunk)))
        else:
            chunk_means.append(0.0)
            chunk_stds.append(0.0)

    features.extend(chunk_means)
    features.extend(chunk_stds)

    chunk_means = np.array(chunk_means, dtype=np.float32)
    chunk_stds = np.array(chunk_stds, dtype=np.float32)

    if len(chunk_means) > 1:
        mean_diffs = np.diff(chunk_means)
        std_diffs = np.diff(chunk_stds)

        features.extend([
            float(np.mean(np.abs(mean_diffs))),
            float(np.std(mean_diffs)),
            float(np.max(mean_diffs)),
            float(np.min(mean_diffs)),
        ])
        features.extend([
            float(np.mean(np.abs(std_diffs))),
            float(np.std(std_diffs)),
            float(np.max(std_diffs)),
            float(np.min(std_diffs)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
        features.extend([0.0, 0.0, 0.0, 0.0])

    features.extend(pe_features.tolist())

    return np.array(features, dtype=np.float32)

class MalwareScanner:
    def __init__(self, lightgbm_model_path, cluster_mapping_path, family_names_path,
                 max_file_size=256*1024, cache_file='scan_cache.json', enable_cache=True):

        self.max_file_size = max_file_size
        self.cache_file = cache_file
        self.enable_cache = enable_cache

        print("[*] Loading LightGBM binary classification model...")
        model_path = lightgbm_model_path
        if model_path.endswith('.gz'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
                with gzip.open(model_path, 'rb') as gf:
                    shutil.copyfileobj(gf, tmp)
                model_path = tmp.name
        self.binary_classifier = lgb.Booster(model_file=model_path)
        self._temp_model_path = model_path if model_path != lightgbm_model_path else None
        print("[+] LightGBM binary classification model loaded")

        print("[*] Loading cluster mapping information...")
        def _load_json_any(path):
            if path.endswith('.gz'):
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        self.cluster_mapping = _load_json_any(cluster_mapping_path)
        print(f"[+] Cluster mapping information loaded, total {len(self.cluster_mapping)} files")

        print("[*] Loading family name mapping...")
        self.family_names = _load_json_any(family_names_path)
        print(f"[+] Family name mapping loaded, total {len(self.family_names)} known families")

        self.scan_cache = self._load_cache()
        if self.enable_cache:
            print(f"[+] Scan cache loaded, total {len(self.scan_cache)} cached files")
        else:
            print("[+] Scan cache disabled for this scanner instance")

        print("[+] Malware scanner initialization completed")

    def _load_cache(self):

        if not self.enable_cache or not self.cache_file:
            return {}

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Failed to load scan cache: {e}")
                return {}
        return {}

    def _save_cache(self):

        if not self.enable_cache or not self.cache_file:
            return

        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.scan_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Failed to save scan cache: {e}")

    def _calculate_sha256(self, file_path):

        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"[!] Failed to calculate SHA256 for file {file_path}: {e}")
            return None

    def _is_pe_file(self, file_path):

        try:
            with open(file_path, 'rb') as f:
                magic = f.read(2)
                if magic != b'MZ':
                    return False

                f.seek(0x3C)
                pe_offset_bytes = f.read(4)
                if len(pe_offset_bytes) < 4:
                    return False

                pe_offset = int.from_bytes(pe_offset_bytes, byteorder='little')

                f.seek(pe_offset)
                pe_signature = f.read(4)
                return pe_signature == b'PE\x00\x00'
        except Exception:
            return False

    def _preprocess_file(self, file_path):

        try:

            byte_sequence, pe_features = extract_features_in_memory(file_path, self.max_file_size)
            if byte_sequence is None or pe_features is None:
                raise Exception("Failed to extract features in memory")

            if len(pe_features) != 1000:
                fixed_pe_features = np.zeros(1000, dtype=np.float32)
                fixed_pe_features[:min(len(pe_features), 1000)] = pe_features[:min(len(pe_features), 1000)]
                pe_features = fixed_pe_features

            features = extract_statistical_features(byte_sequence, pe_features)
            return features
        except Exception as e:
            print(f"[!] File preprocessing failed {file_path}: {e}")
            return None

    def is_malware(self, file_path):

        features = self._preprocess_file(file_path)
        if features is None:
            return False, 0.0

        try:
            feature_vector = features.reshape(1, -1)
            prediction = self.binary_classifier.predict(feature_vector)

            is_malware = prediction[0] > 0.9
            confidence = prediction[0] if is_malware else (1 - prediction[0])

            return is_malware, confidence
        except Exception as e:
            print(f"[!] Binary classification prediction failed {file_path}: {e}")
            return False, 0.0

    def predict_family(self, file_path):

        file_name = os.path.basename(file_path) + '.npz'

        if file_name in self.cluster_mapping:
            cluster_id = self.cluster_mapping[file_name]
            cluster_id_str = str(cluster_id)

            if cluster_id_str in self.family_names:
                family_name = self.family_names[cluster_id_str]
                is_new_family = False
            else:
                family_name = f"New_Family_Cluster_{cluster_id}"
                is_new_family = True

            return cluster_id, family_name, is_new_family
        else:
            return None, "Unknown_Family", True

    def scan_file(self, file_path):

        file_hash = self._calculate_sha256(file_path)
        if file_hash is None:
            print(f"[!] Unable to calculate file hash: {file_path}")
            return None

        if self.enable_cache and file_hash in self.scan_cache:
            cached_result = self.scan_cache[file_hash]
            print(f"[*] Using cached result: {file_path}")
            return cached_result

        if not self._is_pe_file(file_path):
            print(f"[-] Skipping non-PE file: {file_path}")
            return None

        print(f"[*] Scanning file: {file_path}")

        is_malware, confidence = self.is_malware(file_path)

        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'is_malware': bool(is_malware),
            'confidence': float(confidence),
        }

        if is_malware:
            cluster_id, family_name, is_new_family = self.predict_family(file_path)

            result.update({
                'malware_family': {
                    'cluster_id': cluster_id,
                    'family_name': family_name,
                    'is_new_family': bool(is_new_family)
                }
            })

            if is_new_family:
                print(f"[+] New malware family discovered: {family_name}")
            else:
                print(f"[+] Identified as known family: {family_name}")
        else:
            print(f"[+] Identified as benign software")

        if self.enable_cache:
            self.scan_cache[file_hash] = result

        return result

    def scan_directory(self, directory_path, recursive=False):

        results = []

        if recursive:
            files = Path(directory_path).rglob('*')
        else:
            files = Path(directory_path).glob('*')

        files = [f for f in files if f.is_file()]

        print(f"[*] Scanning directory: {directory_path} ({'recursive' if recursive else 'non-recursive'})")
        print(f"[*] Found {len(files)} files")

        for file_path in files:
            try:
                result = self.scan_file(str(file_path))
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"[!] Failed to process file {file_path}: {e}")

        return results

    def save_results(self, results, output_path):

        json_path = output_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[+] Scan results saved to: {json_path}")

        csv_path = output_path + '.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = ['file_path', 'file_name', 'file_size', 'is_malware', 'confidence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    flat_result = {
                        'file_path': result['file_path'],
                        'file_name': result['file_name'],
                        'file_size': result['file_size'],
                        'is_malware': result['is_malware'],
                        'confidence': result['confidence']
                    }
                    writer.writerow(flat_result)
        print(f"[+] Scan results saved to: {csv_path}")

    def __del__(self):

        if getattr(self, 'enable_cache', False):
            self._save_cache()
        try:
            if hasattr(self, '_temp_model_path') and self._temp_model_path:
                os.unlink(self._temp_model_path)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Malware Scanner")

    parser.add_argument('--lightgbm-model-path', type=str,
                       default=os.path.join(BASE_DIR, 'saved_models', 'lightgbm_model.txt'),
                       help='LightGBM binary classification model path')
    parser.add_argument('--cluster-mapping-path', type=str,
                       default=os.path.join(BASE_DIR, 'hdbscan_cluster_results', 'file_cluster_mapping.json'),
                       help='Cluster mapping file path')
    parser.add_argument('--family-names-path', type=str,
                       default=os.path.join(BASE_DIR, 'hdbscan_cluster_results', 'family_names_mapping.json'),
                       help='Family name mapping file path')
    parser.add_argument('--cache-file', type=str, default=os.path.join(BASE_DIR, 'scan_cache.json'),
                       help='Scan cache file path')

    parser.add_argument('--file-path', type=str, help='Single file path')
    parser.add_argument('--dir-path', type=str, help='Directory path')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively scan subdirectories')
    parser.add_argument('--output-path', type=str, default='./scan_results',
                       help='Scan result output path (without extension)')

    parser.add_argument('--max-file-size', type=int, default=256*1024,
                       help='Maximum file size (default: 256KB)')

    args = parser.parse_args()

    if not os.path.exists(args.lightgbm_model_path):
        print(f"[!] Error: LightGBM model file not found {args.lightgbm_model_path}")
        return

    if not os.path.exists(args.cluster_mapping_path):
        print(f"[!] Error: Cluster mapping file not found {args.cluster_mapping_path}")
        return

    if not os.path.exists(args.family_names_path):
        print(f"[!] Error: Family name mapping file not found {args.family_names_path}")
        return

    scanner = MalwareScanner(
        lightgbm_model_path=args.lightgbm_model_path,
        cluster_mapping_path=args.cluster_mapping_path,
        family_names_path=args.family_names_path,
        max_file_size=args.max_file_size,
        cache_file=args.cache_file,
        enable_cache=True,
    )

    results = []
    if args.file_path:
        if not os.path.exists(args.file_path):
            print(f"[!] Error: File does not exist {args.file_path}")
            return
        result = scanner.scan_file(args.file_path)
        if result is not None:
            results.append(result)
    elif args.dir_path:
        if not os.path.exists(args.dir_path):
            print(f"[!] Error: Directory does not exist {args.dir_path}")
            return
        results = scanner.scan_directory(args.dir_path, args.recursive)
    else:
        print("[!] Error: Please specify a file or directory to scan")
        parser.print_help()
        return

    scanner.save_results(results, args.output_path)

    scanner._save_cache()

    malware_count = sum(1 for r in results if r['is_malware'])
    new_family_count = sum(1 for r in results if r['is_malware'] and
                          r.get('malware_family', {}).get('is_new_family', False))

    print(f"\n[*] Scan completion statistics:")
    print(f"    Total files: {len(results)}")
    print(f"    Malware: {malware_count}")
    print(f"    Newly discovered families: {new_family_count}")

if __name__ == '__main__':
    main()