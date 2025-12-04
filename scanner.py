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
import pickle
from sklearn.preprocessing import StandardScaler

from feature_extractor_enhanced import extract_features_in_memory

BASE_DIR = getattr(sys, '_MEIPASS', os.path.abspath('.'))

def validate_path(path):
    if not path:
        return None

    normalized_path = os.path.normpath(path)

    if '\0' in normalized_path:
        return None

    abs_path = os.path.abspath(normalized_path)

    allowed_root = os.getenv('SCANNER_ALLOWED_SCAN_ROOT')
    if allowed_root:
        base = os.path.abspath(allowed_root)
        if not abs_path.startswith(base + os.sep) and abs_path != base:
            return None

    if not os.path.exists(abs_path):
        return None

    return abs_path

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

class FamilyClassifier:
    def __init__(self):
        self.centroids = {}
        self.thresholds = {}
        self.family_names = {}
        self.scaler = None

    def load(self, path):
        if not os.path.exists(path):
            print(f"[!] Classifier model not found: {path}")
            return False
            
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.centroids = data['centroids']
                self.thresholds = data['thresholds']
                self.family_names = data['family_names']
                self.scaler = data.get('scaler')
            print(f"[+] Family classifier loaded, {len(self.centroids)} families")
            return True
        except Exception as e:
            print(f"[!] Failed to load classifier: {e}")
            return False

    def predict(self, feature_vector):
        if not self.centroids:
            return None, "Model_Not_Loaded", True

        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]

        min_dist = float('inf')
        best_label = None

        for label, centroid in self.centroids.items():
            dist = np.linalg.norm(feature_vector - centroid)
            if dist < min_dist:
                min_dist = dist
                best_label = label

        if best_label is not None:
            threshold = self.thresholds[best_label]
            if min_dist <= threshold:
                return best_label, self.family_names[best_label], False
        
        return None, "New_Unknown_Family", True

class MalwareScanner:
    def __init__(self, lightgbm_model_path, family_classifier_path,
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

        print("[*] Loading family classifier...")
        self.family_classifier = FamilyClassifier()
        self.family_classifier.load(family_classifier_path)

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
            valid_path = validate_path(file_path)
            if not valid_path:
                return None

            with open(valid_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"[!] Failed to calculate SHA256 for file {file_path}: {e}")
            return None

    def _is_pe_file(self, file_path):

        try:
            valid_path = validate_path(file_path)
            if not valid_path:
                return False

            with open(valid_path, 'rb') as f:
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

    def _predict_malware_from_features(self, features):
        try:
            feature_vector = features.reshape(1, -1)
            prediction = self.binary_classifier.predict(feature_vector)

            is_malware = prediction[0] > 0.9
            confidence = prediction[0] if is_malware else (1 - prediction[0])

            return is_malware, confidence
        except Exception as e:
            print(f"[!] Binary classification prediction failed: {e}")
            return False, 0.0

    def is_malware(self, file_path):
        features = self._preprocess_file(file_path)
        if features is None:
            return False, 0.0
        
        return self._predict_malware_from_features(features)

    def predict_family(self, features):
        return self.family_classifier.predict(features)

    def scan_file(self, file_path):

        valid_path = validate_path(file_path)
        if not valid_path:
            print(f"[!] Invalid or non-existent file path: {file_path}")
            return None

        file_hash = self._calculate_sha256(valid_path)
        if file_hash is None:
            print(f"[!] Unable to calculate file hash: {valid_path}")
            return None

        if self.enable_cache and file_hash in self.scan_cache:
            cached_result = self.scan_cache[file_hash]
            print(f"[*] Using cached result: {valid_path}")
            return cached_result

        if not self._is_pe_file(valid_path):
            print(f"[-] Skipping non-PE file: {valid_path}")
            return None

        print(f"[*] Scanning file: {valid_path}")

        features = self._preprocess_file(valid_path)
        if features is None:
            return None

        is_malware, confidence = self._predict_malware_from_features(features)

        result = {
            'file_path': valid_path,
            'file_name': os.path.basename(valid_path),
            'file_size': os.path.getsize(valid_path),
            'is_malware': bool(is_malware),
            'confidence': float(confidence),
        }

        if is_malware:
            cluster_id, family_name, is_new_family = self.predict_family(features)

            result.update({
                'malware_family': {
                    'cluster_id': int(cluster_id) if cluster_id is not None else -1,
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
    parser.add_argument('--family-classifier-path', type=str,
                       default=os.path.join(BASE_DIR, 'hdbscan_cluster_results', 'family_classifier.pkl'),
                       help='Family classifier pickle path')
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

    if not os.path.exists(args.family_classifier_path):
        print(f"[!] Error: Family classifier file not found {args.family_classifier_path}")
        return

    scanner = MalwareScanner(
        lightgbm_model_path=args.lightgbm_model_path,
        family_classifier_path=args.family_classifier_path,
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
