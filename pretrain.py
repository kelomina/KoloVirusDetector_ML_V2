import os
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import pickle
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class MalwareDataset:
    def __init__(self, data_dir, file_list, label_list, max_length=256*1024):
        self.data_dir = data_dir
        self.file_list = file_list
        self.label_list = label_list
        self.max_length = max_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.label_list[idx]
        
        if filename.endswith('.npz'):
            file_path = os.path.join(self.data_dir, filename)
        else:
            file_path = os.path.join(self.data_dir, f"{filename}.npz")
        
        try:
            with np.load(file_path) as data:
                byte_sequence = data['byte_sequence']
                if 'pe_features' in data:
                    pe_features = data['pe_features']
                    if pe_features.ndim > 1:
                        pe_features = pe_features.flatten()
                else:
                    pe_features = np.zeros(1000, dtype=np.float32)
        except FileNotFoundError:
            # [Warning] File not found: {file_path}, using zero padding.
            print(f"[Warning] File not found: {file_path}, using zero padding.")
            byte_sequence = np.zeros(self.max_length, dtype=np.uint8)
            pe_features = np.zeros(1000, dtype=np.float32)
        except Exception as e:
            # [Warning] Error reading file {file_path}: {e}, using zero padding.
            print(f"[Warning] Error reading file {file_path}: {e}, using zero padding.")
            byte_sequence = np.zeros(self.max_length, dtype=np.uint8)
            pe_features = np.zeros(1000, dtype=np.float32)
        if len(byte_sequence) > self.max_length:
            byte_sequence = byte_sequence[:self.max_length]
        else:
            byte_sequence = np.pad(byte_sequence, (0, self.max_length - len(byte_sequence)), 'constant')
        if len(pe_features) != 1000:
            fixed_pe_features = np.zeros(1000, dtype=np.float32)
            fixed_pe_features[:min(len(pe_features), 1000)] = pe_features[:min(len(pe_features), 1000)]
            pe_features = fixed_pe_features

        return byte_sequence, pe_features, label

def extract_statistical_features(byte_sequence, pe_features):
    byte_array = np.array(byte_sequence)
    features = []
    features.extend([
        np.mean(byte_array),
        np.std(byte_array),
        np.min(byte_array),
        np.max(byte_array),
        np.median(byte_array),
        np.percentile(byte_array, 25),
        np.percentile(byte_array, 75)
    ])
    features.extend([
        np.sum(byte_array == 0),
        np.sum(byte_array == 0xFF),
        np.sum(byte_array == 0x90),
        np.sum((byte_array >= 32) & (byte_array <= 126))
    ])
    
    hist, _ = np.histogram(byte_array, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
    features.append(entropy)
    
    chunk_size = len(byte_array) // 10
    chunk_means = []
    chunk_stds = []
    for i in range(10):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 9 else len(byte_array)
        chunk = byte_array[start_idx:end_idx]
        chunk_means.append(np.mean(chunk) if len(chunk) > 0 else 0)
        chunk_stds.append(np.std(chunk) if len(chunk) > 0 else 0)
    
    features.extend(chunk_means)
    features.extend(chunk_stds)
    
    features.extend(pe_features.tolist())
    
    return np.array(features, dtype=np.float32)

def load_dataset(data_dir, metadata_file, max_file_size=256*1024, fast_dev_run=False):
    # [*] Loading dataset...
    print("[*] Loading dataset...")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    label_map = {}
    for file, label in metadata.items():
        if label == 'benign':
            label_map[file] = 0
        elif label == 'malicious':
            label_map[file] = 1
        else:
            # 1 if label != '待加入白名单' else 0
            label_map[file] = 1 if label != '待加入白名单' else 0
    
    all_files = list(metadata.keys())
    all_labels = [label_map[fname] for fname in all_files]
    
    if fast_dev_run:
        # [!] Fast development mode enabled, balancing benign and malicious samples.
        print("[!] Fast development mode enabled, balancing benign and malicious samples.")
        benign_files = [f for f, label in zip(all_files, all_labels) if label == 0]
        malicious_files = [f for f, label in zip(all_files, all_labels) if label == 1]
        
        n_samples_per_class = 5000
        selected_benign_files = benign_files[:min(n_samples_per_class, len(benign_files))]
        selected_malicious_files = malicious_files[:min(n_samples_per_class, len(malicious_files))]
        
        all_files = selected_benign_files + selected_malicious_files
        all_labels = [0] * len(selected_benign_files) + [1] * len(selected_malicious_files)
        
        # Benign samples: {len(selected_benign_files)}
        print(f"    Benign samples: {len(selected_benign_files)}")
        # Malicious samples: {len(selected_malicious_files)}
        print(f"    Malicious samples: {len(selected_malicious_files)}")
    
    # [+] Loaded {len(all_files)} files
    print(f"[+] Loaded {len(all_files)} files")
    
    features_list = []
    labels_list = []
    valid_files = []
    
    dataset = MalwareDataset(data_dir, all_files, all_labels, max_file_size)
    
    total_samples = len(dataset)
    progress_desc = "Extracting features"
    
    for i in tqdm(range(total_samples), desc=progress_desc):
        try:
            byte_sequence, pe_features, label = dataset[i]
            features = extract_statistical_features(byte_sequence, pe_features)
            features_list.append(features)
            labels_list.append(label)
            valid_files.append(all_files[i])
        except Exception as e:
            # [!] Error processing file {all_files[i]}: {e}
            print(f"[!] Error processing file {all_files[i]}: {e}")
            continue
    
    try:
        X = np.array(features_list, dtype=np.float32)
    except ValueError as e:
        # [!] Feature array shape inconsistency: {e}
        print(f"[!] Feature array shape inconsistency: {e}")
        # [*] Attempting to manually align feature dimensions...
        print("[*] Attempting to manually align feature dimensions...")
        max_features = max(len(f) for f in features_list)
        aligned_features = []
        for f in features_list:
            if len(f) < max_features:
                padded_f = np.zeros(max_features, dtype=np.float32)
                padded_f[:len(f)] = f
                aligned_features.append(padded_f)
            else:
                aligned_features.append(f)
        X = np.array(aligned_features, dtype=np.float32)
    
    y = np.array(labels_list)
    
    # [+] Feature extraction completed, feature dimension: {X.shape[1]}
    print(f"[+] Feature extraction completed, feature dimension: {X.shape[1]}")
    # [+] Valid samples: {X.shape[0]}
    print(f"[+] Valid samples: {X.shape[0]}")
    
    return X, y, valid_files

def extract_features_from_raw_files(data_dir, output_dir, max_file_size=256*1024, 
                                  file_extensions=None, label_inference='filename'):
    # [*] Extracting features from raw files: {data_dir}
    print(f"[*] Extracting features from raw files: {data_dir}")
    # [*] Output directory: {output_dir}
    print(f"[*] Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file_extensions:
                _, ext = os.path.splitext(file)
                if ext.lower() not in file_extensions:
                    continue
                    
            all_files.append(file_path)
    
    if not all_files:
        # [!] No files found in raw file directory: {data_dir}
        print(f"[!] No files found in raw file directory: {data_dir}")
        return [], []
    
    # [+] Found {len(all_files)} files in raw file directory
    print(f"[+] Found {len(all_files)} files in raw file directory")
    
    try:
        from feature_extractor_enhanced import process_file_directory
        # [+] Successfully imported feature extraction module
        print("[+] Successfully imported feature extraction module")
    except ImportError as e:
        # [!] Failed to import feature extraction module: {e}
        print(f"[!] Failed to import feature extraction module: {e}")
        return [], []
    
    labels = []
    output_files = []
    
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, data_dir)
        output_file = os.path.join(output_dir, rel_path + '.npz')
        output_files.append(output_file)
        
        output_subdir = os.path.dirname(output_file)
        os.makedirs(output_subdir, exist_ok=True)
        
        if label_inference == 'filename':
            file_name = os.path.basename(file_path)
            # Check if file name contains keywords indicating benign files
            if 'benign' in file_name.lower() or 'good' in file_name.lower() or 'clean' in file_name.lower():
                labels.append(0)
            else:
                labels.append(1)
        elif label_inference == 'directory':
            parent_dir = os.path.basename(os.path.dirname(file_path))
            # Check if directory name contains keywords indicating benign files
            if 'benign' in parent_dir.lower() or 'good' in parent_dir.lower() or 'clean' in parent_dir.lower():
                labels.append(0)
            else:
                labels.append(1)
        else:
            labels.append(1)
    
    # [*] Starting feature extraction...
    print("[*] Starting feature extraction...")
    success_count = 0
    
    for i, (input_file, output_file) in enumerate(tqdm(zip(all_files, output_files), 
                                                       total=len(all_files), 
                                                       desc="Feature extraction")):
        try:
            process_file_directory(input_file, output_file, max_file_size)
            success_count += 1
        except Exception as e:
            # [!] Error processing file {input_file}: {e}
            print(f"[!] Error processing file {input_file}: {e}")
            if output_file in output_files:
                idx = output_files.index(output_file)
                output_files.pop(idx)
                labels.pop(idx)
    
    # [+] Feature extraction completed: {success_count}/{len(all_files)} files processed successfully
    print(f"[+] Feature extraction completed: {success_count}/{len(all_files)} files processed successfully")
    
    file_names = [os.path.basename(f) for f in output_files]
    return file_names, labels

def load_incremental_dataset(data_dir, max_file_size=256*1024):
    # [*] Loading dataset from incremental directory: {data_dir}
    print(f"[*] Loading dataset from incremental directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        # [!] Incremental training directory does not exist: {data_dir}
        print(f"[!] Incremental training directory does not exist: {data_dir}")
        return None, None, None
    
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        # [!] No .npz files found in incremental training directory: {data_dir}
        print(f"[!] No .npz files found in incremental training directory: {data_dir}")
        return None, None, None
    
    # [+] Found {len(all_files)} files in incremental directory
    print(f"[+] Found {len(all_files)} files in incremental directory")
    
    labels = []
    valid_files = []
    file_names = []
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # Check if file name contains keywords indicating benign files
        if 'benign' in file_name.lower() or 'good' in file_name.lower() or 'clean' in file_name.lower():
            labels.append(0)
        else:
            labels.append(1)
            
        valid_files.append(file_path)
        file_names.append(file_name)
    
    features_list = []
    valid_file_names = []
    
    for i, file_path in enumerate(tqdm(valid_files, desc="Extracting incremental features")):
        try:
            with np.load(file_path) as data:
                byte_sequence = data['byte_sequence']
                if 'pe_features' in data:
                    pe_features = data['pe_features']
                    if pe_features.ndim > 1:
                        pe_features = pe_features.flatten()
                else:
                    pe_features = np.zeros(1000, dtype=np.float32)
            
            if len(byte_sequence) > max_file_size:
                byte_sequence = byte_sequence[:max_file_size]
            else:
                byte_sequence = np.pad(byte_sequence, (0, max_file_size - len(byte_sequence)), 'constant')

            if len(pe_features) != 1000:
                fixed_pe_features = np.zeros(1000, dtype=np.float32)
                fixed_pe_features[:min(len(pe_features), 1000)] = pe_features[:min(len(pe_features), 1000)]
                pe_features = fixed_pe_features

            features = extract_statistical_features(byte_sequence, pe_features)
            features_list.append(features)
            valid_file_names.append(file_names[i])
        except Exception as e:
            # [!] Error processing file {file_path}: {e}
            print(f"[!] Error processing file {file_path}: {e}")
            continue
    
    if not features_list:
        # [!] Failed to extract any features from incremental data
        print("[!] Failed to extract any features from incremental data")
        return None, None, None
    
    try:
        X = np.array(features_list, dtype=np.float32)
    except ValueError as e:
        # [!] Incremental feature array shape inconsistency: {e}
        print(f"[!] Incremental feature array shape inconsistency: {e}")
        return None, None, None
    
    y = np.array(labels[:len(features_list)])
    
    # [+] Incremental feature extraction completed, feature dimension: {X.shape[1]}
    print(f"[+] Incremental feature extraction completed, feature dimension: {X.shape[1]}")
    # [+] Valid samples: {X.shape[0]}
    print(f"[+] Valid samples: {X.shape[0]}")
    
    return X, y, valid_file_names

def save_features(X, y, files, save_dir):
    # [*] Saving features to file...
    print("[*] Saving features to file...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    features_path = os.path.join(save_dir, 'features.npz')
    np.savez_compressed(features_path, X=X, y=y, files=files)
    
    csv_path = os.path.join(save_dir, 'features.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['filename', 'label'] + [f'feature_{i}' for i in range(X.shape[1])]
        writer.writerow(header)
        for i in range(X.shape[0]):
            row = [files[i], y[i]] + X[i].tolist()
            writer.writerow(row)
    
    # [+] Features saved to: {features_path}
    print(f"[+] Features saved to: {features_path}")
    # [+] CSV format features saved to: {csv_path}
    print(f"[+] CSV format features saved to: {csv_path}")

def train_lightgbm_model(X_train, y_train, X_val, y_val, false_positive_files=None, files_train=None, iteration=1):
    # [*] Training LightGBM model (Round {iteration})...
    print(f"[*] Training LightGBM model (Round {iteration})...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    if false_positive_files is not None and files_train is not None:
        # [*] Detected {len(false_positive_files)} false positive samples, increasing their training weights
        print(f"[*] Detected {len(false_positive_files)} false positive samples, increasing their training weights")
        weights = np.ones(len(X_train), dtype=np.float32)
        false_positive_count = 0
        
        # [*] Current false positive weight factor: {weight_factor}
        weight_factor = min(5.0 + iteration * 2.0, 50.0)
        print(f"[*] Current false positive weight factor: {weight_factor}")
        
        for i, file in enumerate(files_train):
            if file in false_positive_files:
                weights[i] = weight_factor
                false_positive_count += 1
        
        # [+] Identified {false_positive_count} false positive samples, adjusted weights
        print(f"[+] Identified {false_positive_count} false positive samples, adjusted weights")
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    learning_rate = max(0.05 / (1.0 + iteration * 0.1), 0.01)
    num_leaves = min(31 + iteration * 5, 128)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_gain_to_split': 0.01,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), 8)
    }
    
    # [*] Current training parameters - Learning rate: {learning_rate:.4f}, Number of leaves: {num_leaves}
    print(f"[*] Current training parameters - Learning rate: {learning_rate:.4f}, Number of leaves: {num_leaves}")
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['validation'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    return model

def evaluate_model(model, X_test, y_test, files_test=None):
    # [*] Evaluating model...
    print("[*] Evaluating model...")
    
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    # [+] Accuracy: {accuracy:.4f}
    print(f"[+] Accuracy: {accuracy:.4f}")
    
    # [*] Classification report:
    print("\n[*] Classification report:")
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = []
    if 0 in unique_labels:
        target_names.append('Benign')
    if 1 in unique_labels:
        target_names.append('Malicious')
    
    if len(unique_labels) > 1:
        print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels))
    else:
        label_name = 'Benign' if unique_labels[0] == 0 else 'Malicious'
        # All samples in test set belong to '{label_name}' category
        print(f"All samples in test set belong to '{label_name}' category")
        precision = accuracy_score(y_test, y_pred)
        # Precision: {precision:.4f}
        print(f"Precision: {precision:.4f}")
    
    false_positives = []
    if files_test is not None:
        fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
        false_positives = [files_test[i] for i in fp_indices]
        # [*] Detected {len(false_positives)} false positive samples:
        print(f"\n[*] Detected {len(false_positives)} false positive samples:")
        for fp_file in false_positives[:10]:
            print(f"    - {fp_file}")
        if len(false_positives) > 10:
            # ... and {len(false_positives) - 10} more false positive samples
            print(f"    ... and {len(false_positives) - 10} more false positive samples")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(unique_labels) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                    xticklabels=target_names, 
                    yticklabels=target_names)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        if 0 in unique_labels:
            ax2.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Benign', color='blue')
        if 1 in unique_labels:
            ax2.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Malicious', color='red')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('Prediction Probability Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # [+] Evaluation charts saved as model_evaluation.png
        print("[+] Evaluation charts saved as model_evaluation.png")
    else:
        # [*] Skipping visualization chart generation as test set contains only one category
        print("[*] Skipping visualization chart generation as test set contains only one category")
    
    return accuracy, false_positives

def save_model(model, model_path):
    model.save_model(model_path)
    # [+] Model saved to: {model_path}
    print(f"[+] Model saved to: {model_path}")

def load_existing_model(model_path):
    if os.path.exists(model_path):
        # [*] Loading existing model: {model_path}
        print(f"[*] Loading existing model: {model_path}")
        try:
            model = lgb.Booster(model_file=model_path)
            # [+] Existing model loaded successfully
            print("[+] Existing model loaded successfully")
            return model
        except Exception as e:
            # [!] Model loading failed: {e}
            print(f"[!] Model loading failed: {e}")
            return None
    else:
        # [-] Existing model not found: {model_path}
        print(f"[-] Existing model not found: {model_path}")
        return None

def incremental_train_lightgbm_model(existing_model, X_train, y_train, X_val, y_val, 
                                   false_positive_files=None, files_train=None,
                                   num_boost_round=100, early_stopping_rounds=50):
    # [*] Performing incremental reinforcement training...
    print("[*] Performing incremental reinforcement training...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    if false_positive_files is not None and files_train is not None:
        # [*] Detected {len(false_positive_files)} false positive samples, increasing their training weights
        print(f"[*] Detected {len(false_positive_files)} false positive samples, increasing their training weights")
        weights = np.ones(len(X_train), dtype=np.float32)
        false_positive_count = 0
        
        for i, file in enumerate(files_train):
            if file in false_positive_files:
                weights[i] = 10.0
                false_positive_count += 1
        
        # [+] Identified {false_positive_count} false positive samples, adjusted weights
        print(f"[+] Identified {false_positive_count} false positive samples, adjusted weights")
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = existing_model.params if existing_model else {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_gain_to_split': 0.01,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), 8)
    }
    
    if existing_model:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['validation'],
            num_boost_round=num_boost_round,
            init_model=existing_model,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(10)]
        )
    else:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['validation'],
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(10)]
        )
    
    # [+] Incremental reinforcement training completed
    print("[+] Incremental reinforcement training completed")
    return model

def save_features_to_csv(X, y, files, output_path):
    # [*] Saving features to {output_path}...
    print(f"[*] Saving features to {output_path}...")
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    df_data = { 'filename': files, 'label': y }
    for i, feature_name in enumerate(feature_names):
        df_data[feature_name] = X[:, i]
    
    df = pd.DataFrame(df_data)
    
    df.to_csv(output_path, index=False)
    # [+] Features saved to: {output_path}
    print(f"[+] Features saved to: {output_path}")

def save_features_to_pickle(X, y, files, output_path):
    # [*] Saving features to {output_path}...
    print(f"[*] Saving features to {output_path}...")
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    df_data = { 'filename': files, 'label': y }
    for i, feature_name in enumerate(feature_names):
        df_data[feature_name] = X[:, i]
    
    df = pd.DataFrame(df_data)
    
    df.to_pickle(output_path)
    # [+] Features saved to: {output_path}
    print(f"[+] Features saved to: {output_path}")

def main(args):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_lightgbm')
    METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'metadata.json')
    SAVED_MODEL_PATH = os.path.join(BASE_DIR, 'saved_models')
    os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
    
    MODEL_PATH = os.path.join(SAVED_MODEL_PATH, 'lightgbm_model.txt')
    
    FEATURES_PKL_PATH = os.path.join(BASE_DIR, 'extracted_features.pkl')
    
    if args.use_existing_features and os.path.exists(FEATURES_PKL_PATH):
        # [*] Loading existing feature file...
        print("[*] Loading existing feature file...")
        try:
            df = pd.read_pickle(FEATURES_PKL_PATH)
            files = df['filename'].tolist()
            y = df['label'].values
            X = df.drop(['filename', 'label'], axis=1).values
            
            # [+] Successfully loaded feature file, total {len(files)} samples, feature dimension: {X.shape[1]}
            print(f"[+] Successfully loaded feature file, total {len(files)} samples, feature dimension: {X.shape[1]}")
        except Exception as e:
            # [!] Failed to load feature file: {e}
            print(f"[!] Failed to load feature file: {e}")
            # [-] Exiting training
            print("[-] Exiting training")
            return
    else:
        if args.incremental_training and args.incremental_data_dir:
            if args.incremental_raw_data_dir:
                # [*] Extracting features from raw files...
                print("[*] Extracting features from raw files...")
                output_features_dir = args.incremental_data_dir
                file_names, labels = extract_features_from_raw_files(
                    args.incremental_raw_data_dir,
                    output_features_dir,
                    args.max_file_size,
                    args.file_extensions,
                    args.label_inference
                )
                
                if not file_names:
                    # [!] Failed to extract features from raw files, exiting training
                    print("[!] Failed to extract features from raw files, exiting training")
                    return
                    
            X, y, files = load_incremental_dataset(args.incremental_data_dir, args.max_file_size)
            if X is None:
                # [!] Failed to load incremental data, exiting training
                print("[!] Failed to load incremental data, exiting training")
                return
        else:
            X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, args.max_file_size, args.fast_dev_run)
        
        save_features_to_pickle(X, y, files, FEATURES_PKL_PATH)
    
    if len(X) > 10:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        if len(X_temp) > 5:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
            )
        else:
            X_train, X_val = X_temp, X_temp
            y_train, y_val = y_temp, y_temp
            X_test, y_test = X_temp, y_temp
    else:
        X_train, X_val, X_test = X, X, X
        y_train, y_val, y_test = y, y, y
    
    files_train = [f"file_{i}.npz" for i in range(len(X_train))]
    files_val = [f"file_{i}.npz" for i in range(len(X_val))]
    files_test = [f"file_{i}.npz" for i in range(len(X_test))]
    
    # [*] Dataset split completed:
    print(f"[*] Dataset split completed:")
    # Training set: {len(X_train)} samples
    print(f"    Training set: {len(X_train)} samples")
    # Validation set: {len(X_val)} samples
    print(f"    Validation set: {len(X_val)} samples")
    # Test set: {len(X_test)} samples
    print(f"    Test set: {len(X_test)} samples")
    
    existing_model = None
    if args.incremental_training:
        existing_model = load_existing_model(MODEL_PATH)
    
    model = None
    
    if args.incremental_training and existing_model:
        # [*] Performing incremental training...
        print("\n[*] Performing incremental training...")
        model = incremental_train_lightgbm_model(
            existing_model, X_train, y_train, X_val, y_val,
            num_boost_round=args.incremental_rounds,
            early_stopping_rounds=args.incremental_early_stopping
        )
    else:
        model = train_lightgbm_model(X_train, y_train, X_val, y_val, iteration=1)
    
    max_finetune_iterations = args.max_finetune_iterations
    finetune_iteration = 0
    false_positives = []
    previous_false_positives = set()
    stuck_counter = 0
    
    while finetune_iteration < max_finetune_iterations:
        if args.finetune_on_false_positives:
            finetune_iteration += 1
            # [*] Performing round {finetune_iteration} reinforcement training...
            print(f"\n[*] Performing round {finetune_iteration} reinforcement training...")
            model = train_lightgbm_model(X_train, y_train, X_val, y_val, iteration=finetune_iteration+1)
            
            if finetune_iteration >= max_finetune_iterations:
                # [*] Reached maximum reinforcement training rounds
                print("[*] Reached maximum reinforcement training rounds")
                break
        else:
            # [*] Reinforcement training not enabled, skipping reinforcement training phase
            print("[*] Reinforcement training not enabled, skipping reinforcement training phase")
            break
    
    # [*] Reinforcement training completed, performing final evaluation...
    print("\n[*] Reinforcement training completed, performing final evaluation...")
    if len(X_test) > 0:
        test_accuracy, false_positives = evaluate_model(model, X_test, y_test, files_test)
        
        if false_positives and args.finetune_on_false_positives:
            # [*] Detected {len(false_positives)} false positive samples, performing targeted reinforcement training...
            print(f"\n[*] Detected {len(false_positives)} false positive samples, performing targeted reinforcement training...")
            
            targeted_iteration = 0
            max_targeted_iterations = 5
            previous_fp_count = len(false_positives)
            
            while len(false_positives) > 0 and targeted_iteration < max_targeted_iterations:
                targeted_iteration += 1
                # [*] Performing round {targeted_iteration} targeted reinforcement training...
                print(f"\n[*] Performing round {targeted_iteration} targeted reinforcement training...")
                model = train_lightgbm_model(X_train, y_train, X_val, y_val, 
                                           false_positives, files_train, 
                                           finetune_iteration + targeted_iteration)
                
                # [*] Evaluating after round {targeted_iteration} targeted reinforcement training...
                print(f"\n[*] Evaluating after round {targeted_iteration} targeted reinforcement training...")
                test_accuracy, false_positives = evaluate_model(model, X_test, y_test, files_test)
                
                if len(false_positives) >= previous_fp_count:
                    # [*] Targeted reinforcement training failed to reduce false positives, stopping training
                    print("[*] Targeted reinforcement training failed to reduce false positives, stopping training")
                    break
                previous_fp_count = len(false_positives)
                
            if len(false_positives) == 0:
                # [*] Successfully eliminated all false positive samples
                print("[*] Successfully eliminated all false positive samples")
            else:
                # [*] Targeted reinforcement training completed, remaining {len(false_positives)} false positive samples
                print(f"[*] Targeted reinforcement training completed, remaining {len(false_positives)} false positive samples")
        elif false_positives:
            # [*] Detected {len(false_positives)} false positive samples, but reinforcement training is not enabled
            print(f"\n[*] Detected {len(false_positives)} false positive samples, but reinforcement training is not enabled")
            # To enable reinforcement training, use the --finetune-on-false-positives parameter
            print("    To enable reinforcement training, use the --finetune-on-false-positives parameter")
    else:
        # [*] Test set is empty, skipping model evaluation
        print("[*] Test set is empty, skipping model evaluation")
    
    save_model(model, MODEL_PATH)
    
    # [*] Top 20 important features:
    print("\n[*] Top 20 important features:")
    feature_importance = model.feature_importance(importance_type='gain')
    feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(importance_pairs[:20]):
        # {i+1:2d}. {name}: {importance:.2f}
        print(f"    {i+1:2d}. {name}: {importance:.2f}")
    
    # [+] LightGBM pre-training completed! Model saved to: {MODEL_PATH}
    print(f"\n[+] LightGBM pre-training completed! Model saved to: {MODEL_PATH}")
    # [+] Extracted features saved to: {FEATURES_PKL_PATH}
    print(f"[+] Extracted features saved to: {FEATURES_PKL_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LightGBM-based malware detection pre-training script")
    parser.add_argument('--max-file-size', type=int, default=256 * 1024, help='Maximum file size in bytes to process')
    parser.add_argument('--fast-dev-run', action='store_true', help='Use a small portion of data for quick development testing')
    parser.add_argument('--save-features', action='store_true', help='Save extracted features to file')
    parser.add_argument('--finetune-on-false-positives', action='store_true', 
                        help='Perform reinforcement training when false positive samples are detected')
    parser.add_argument('--incremental-training', action='store_true',
                        help='Enable incremental training (continue training based on existing model)')
    parser.add_argument('--incremental-data-dir', type=str, 
                        help='Incremental training data directory (.npz files)')
    parser.add_argument('--incremental-raw-data-dir', type=str,
                        help='Incremental training raw data directory (for feature extraction)')
    parser.add_argument('--file-extensions', type=str, nargs='+',   
                        help='File extensions to process, e.g. .exe .dll')
    parser.add_argument('--label-inference', type=str, default='filename',
                        choices=['filename', 'directory'],
                        help='Label inference method: filename (based on file name) or directory (based on directory name)')
    parser.add_argument('--incremental-rounds', type=int, default=100,
                        help='Number of rounds for incremental training (default: 100)')
    parser.add_argument('--incremental-early-stopping', type=int, default=50,
                        help='Early stopping rounds for incremental training (default: 50)')
    parser.add_argument('--max-finetune-iterations', type=int, default=10,
                        help='Maximum reinforcement training iterations (default: 10)')
    parser.add_argument('--use-existing-features', action='store_true',
                        help='Use existing extracted_features.pkl file, skip feature extraction')
    
    args = parser.parse_args()
    
    if args.incremental_training and not args.incremental_data_dir:
        # [!] --incremental-data-dir parameter must be specified when enabling incremental training
        print("[!] --incremental-data-dir parameter must be specified when enabling incremental training")
        exit(1)
    
    if args.incremental_raw_data_dir and not args.incremental_data_dir:
        # [!] --incremental-data-dir parameter must be specified when specifying --incremental-raw-data-dir
        print("[!] --incremental-data-dir parameter must be specified when specifying --incremental-raw-data-dir")
        exit(1)
    
    main(args)