import os
import json
import argparse
import numpy as np
import lightgbm as lgb
from pathlib import Path
import csv
import hashlib

from pretrain import extract_statistical_features

class MalwareScanner:
    def __init__(self, lightgbm_model_path, cluster_mapping_path, family_names_path, 
                 max_file_size=256*1024, cache_file='scan_cache.json'):
        """
        Initialize the malware scanner
        
        Args:
            lightgbm_model_path (str): Path to the LightGBM binary classification model
            cluster_mapping_path (str): Path to the cluster mapping file
            family_names_path (str): Path to the family names mapping file
            max_file_size (int): Maximum file size to process
            cache_file (str): Path to the scan cache file
        """
        
        self.max_file_size = max_file_size
        self.cache_file = cache_file
        
        print("[*] Loading LightGBM binary classification model...")
        self.binary_classifier = lgb.Booster(model_file=lightgbm_model_path)
        print("[+] LightGBM binary classification model loaded")
        
        print("[*] Loading cluster mapping information...")
        with open(cluster_mapping_path, 'r', encoding='utf-8') as f:
            self.cluster_mapping = json.load(f)
        print(f"[+] Cluster mapping information loaded, total {len(self.cluster_mapping)} files")
        
        print("[*] Loading family name mapping...")
        with open(family_names_path, 'r', encoding='utf-8') as f:
            self.family_names = json.load(f)
        print(f"[+] Family name mapping loaded, total {len(self.family_names)} known families")
        
        self.scan_cache = self._load_cache()
        print(f"[+] Scan cache loaded, total {len(self.scan_cache)} cached files")
        
        print("[+] Malware scanner initialization completed")

    def _load_cache(self):
        """
        Load scan cache from file
        
        Returns:
            dict: Scan cache data
        """
        
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Failed to load scan cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """
        Save scan cache to file
        """
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.scan_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Failed to save scan cache: {e}")

    def _calculate_sha256(self, file_path):
        """
        Calculate SHA256 hash of a file
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: SHA256 hash or None if failed
        """
        
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
        """
        Check if a file is a PE (Portable Executable) file
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if it's a PE file, False otherwise
        """
        
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
        """
        Preprocess file for feature extraction
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            numpy.ndarray: Extracted features or None if failed
        """
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read(self.max_file_size)
            
            byte_sequence = np.frombuffer(content, dtype=np.uint8)
            
            if len(byte_sequence) > self.max_file_size:
                byte_sequence = byte_sequence[:self.max_file_size]
            else:
                byte_sequence = np.pad(byte_sequence, (0, self.max_file_size - len(byte_sequence)), 'constant')
            
            pe_features = np.zeros(1000, dtype=np.float32)
            
            features = extract_statistical_features(byte_sequence, pe_features)
            
            return features
        except Exception as e:
            print(f"[!] File preprocessing failed {file_path}: {e}")
            return None

    def is_malware(self, file_path):
        """
        Determine if a file is malware using the binary classifier
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            tuple: (is_malware, confidence) or (False, 0.0) if failed
        """
        
        features = self._preprocess_file(file_path)
        if features is None:
            return False, 0.0
            
        try:
            feature_vector = features.reshape(1, -1)
            prediction = self.binary_classifier.predict(feature_vector)
            
            is_malware = prediction[0] > 0.5
            confidence = prediction[0] if is_malware else (1 - prediction[0])
            
            return is_malware, confidence
        except Exception as e:
            print(f"[!] Binary classification prediction failed {file_path}: {e}")
            return False, 0.0

    def predict_family(self, file_path):
        """
        Predict the malware family of a file
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            tuple: (cluster_id, family_name, is_new_family)
        """
        
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
        """
        Scan a single file for malware
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: Scan result or None if failed
        """
        
        file_hash = self._calculate_sha256(file_path)
        if file_hash is None:
            print(f"[!] Unable to calculate file hash: {file_path}")
            return None
        
        if file_hash in self.scan_cache:
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
        
        self.scan_cache[file_hash] = result
        
        return result

    def scan_directory(self, directory_path, recursive=False):
        """
        Scan all files in a directory
        
        Args:
            directory_path (str): Path to the directory
            recursive (bool): Whether to scan subdirectories recursively
            
        Returns:
            list: List of scan results
        """
        
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
        """
        Save scan results to JSON and CSV files
        
        Args:
            results (list): List of scan results
            output_path (str): Base path for output files (without extension)
        """
        
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
        """
        Destructor - save cache when object is destroyed
        """
        
        self._save_cache()


def main():
    parser = argparse.ArgumentParser(description="Malware Scanner")
    
    parser.add_argument('--lightgbm-model-path', type=str, 
                       default=os.path.join('.', 'saved_models', 'lightgbm_model.txt'),
                       help='LightGBM binary classification model path')
    parser.add_argument('--cluster-mapping-path', type=str,
                       default=os.path.join('.', 'hdbscan_cluster_results', 'file_cluster_mapping.json'),
                       help='Cluster mapping file path')
    parser.add_argument('--family-names-path', type=str,
                       default=os.path.join('.', 'hdbscan_cluster_results', 'family_names_mapping.json'),
                       help='Family name mapping file path')
    parser.add_argument('--cache-file', type=str, default='./scan_cache.json',
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
        cache_file=args.cache_file
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