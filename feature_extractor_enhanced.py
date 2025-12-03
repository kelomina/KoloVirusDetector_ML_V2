import numpy as np
import pefile
import os
import sys
import json
import multiprocessing
import hashlib
import struct
from datetime import datetime

MAX_FILE_SIZE =  256 * 1024
PE_FEATURE_SIZE = 500

ERROR_COUNT = 0

def increment_error():
    global ERROR_COUNT
    ERROR_COUNT += 1

def extract_byte_sequence(file_path):
    try:
        with open(file_path, 'rb') as f:
            f.seek(8)
            byte_sequence = np.fromfile(f, dtype=np.uint8, count=MAX_FILE_SIZE - 8)

        if len(byte_sequence) < MAX_FILE_SIZE - 8:
            padded_sequence = np.zeros(MAX_FILE_SIZE, dtype=np.uint8)
            padded_sequence[:len(byte_sequence)] = byte_sequence
            return padded_sequence

        full_sequence = np.zeros(MAX_FILE_SIZE, dtype=np.uint8)
        full_sequence[:len(byte_sequence)] = byte_sequence
        return full_sequence
    except Exception:
        increment_error()
        return None

def calculate_byte_entropy(byte_sequence, block_size=1024):
    if byte_sequence is None or len(byte_sequence) == 0:
        return 0, 0, 0, [], 0

    hist = np.bincount(byte_sequence, minlength=256)
    prob = hist / len(byte_sequence)
    prob = prob[prob > 0]
    overall_entropy = -np.sum(prob * np.log2(prob)) / 8

    block_entropies = []

    num_blocks = min(10, max(1, len(byte_sequence) // block_size))
    if num_blocks > 1:
        block_size = len(byte_sequence) // num_blocks
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size if i < num_blocks - 1 else len(byte_sequence)
            block = byte_sequence[start_idx:end_idx]
            if len(block) > 0:
                block_hist = np.bincount(block, minlength=256)
                block_prob = block_hist / len(block)
                block_prob = block_prob[block_prob > 0]
                if len(block_prob) > 0:
                    block_entropy = -np.sum(block_prob * np.log2(block_prob)) / 8
                    block_entropies.append(block_entropy)
    else:

        block = byte_sequence
        if len(block) > 0:
            block_hist = np.bincount(block, minlength=256)
            block_prob = block_hist / len(block)
            block_prob = block_prob[block_prob > 0]
            if len(block_prob) > 0:
                block_entropy = -np.sum(block_prob * np.log2(block_prob)) / 8
                block_entropies.append(block_entropy)

    if block_entropies:
        return overall_entropy, np.min(block_entropies), np.max(block_entropies), block_entropies, np.std(block_entropies)
    else:
        return overall_entropy, overall_entropy, overall_entropy, [], 0

def extract_file_attributes(file_path):
    features = {}
    missing_flags = {}

    try:
        stat = os.stat(file_path)
        features['size'] = stat.st_size
        missing_flags['size_missing'] = 0

        features['log_size'] = np.log(stat.st_size + 1)
        missing_flags['log_size_missing'] = 0

        with open(file_path, 'rb') as f:
            sample_data = np.fromfile(f, dtype=np.uint8, count=10240)

        avg_entropy, min_entropy, max_entropy, block_entropies, entropy_std = calculate_byte_entropy(sample_data)
        features['file_entropy_avg'] = avg_entropy
        features['file_entropy_min'] = min_entropy
        features['file_entropy_max'] = max_entropy
        features['file_entropy_range'] = max_entropy - min_entropy
        features['file_entropy_std'] = entropy_std
        missing_flags['file_entropy_missing'] = 0

        if block_entropies:
            features['file_entropy_q25'] = np.percentile(block_entropies, 25)
            features['file_entropy_q75'] = np.percentile(block_entropies, 75)
            features['file_entropy_median'] = np.median(block_entropies)
            missing_flags['file_entropy_percentiles_missing'] = 0

            high_entropy_count = sum(1 for e in block_entropies if e > 0.8)
            features['high_entropy_ratio'] = high_entropy_count / len(block_entropies)

            low_entropy_count = sum(1 for e in block_entropies if e < 0.2)
            features['low_entropy_ratio'] = low_entropy_count / len(block_entropies)

            if len(block_entropies) > 1:
                entropy_changes = np.diff(block_entropies)
                features['entropy_change_rate'] = np.mean(np.abs(entropy_changes))
                features['entropy_change_std'] = np.std(entropy_changes)
            else:
                features['entropy_change_rate'] = 0
                features['entropy_change_std'] = 0
        else:
            features['file_entropy_q25'] = 0
            features['file_entropy_q75'] = 0
            features['file_entropy_median'] = 0
            features['high_entropy_ratio'] = 0
            features['low_entropy_ratio'] = 0
            features['entropy_change_rate'] = 0
            features['entropy_change_std'] = 0
            missing_flags['file_entropy_percentiles_missing'] = 1

        if len(sample_data) > 0:
            zero_ratio = np.sum(sample_data == 0) / len(sample_data)
            printable_ratio = np.sum((sample_data >= 32) & (sample_data <= 126)) / len(sample_data)
            features['zero_byte_ratio'] = zero_ratio
            features['printable_byte_ratio'] = printable_ratio
            missing_flags['byte_stats_missing'] = 0
        else:
            features['zero_byte_ratio'] = 0
            features['printable_byte_ratio'] = 0
            missing_flags['byte_stats_missing'] = 1

    except Exception:
        increment_error()

        feature_names = ['size', 'log_size', 'file_entropy_avg', 'file_entropy_min', 'file_entropy_max',
                        'file_entropy_range', 'file_entropy_std', 'file_entropy_q25', 'file_entropy_q75',
                        'file_entropy_median', 'high_entropy_ratio', 'low_entropy_ratio', 'entropy_change_rate',
                        'entropy_change_std', 'zero_byte_ratio', 'printable_byte_ratio']
        for name in feature_names:
            features[name] = 0
        missing_flags['size_missing'] = 1
        missing_flags['log_size_missing'] = 1
        missing_flags['file_entropy_missing'] = 1
        missing_flags['file_entropy_percentiles_missing'] = 1
        missing_flags['byte_stats_missing'] = 1

    features.update(missing_flags)
    return features

def extract_enhanced_pe_features(file_path):
    features = {}
    missing_flags = {}

    try:
        pe = pefile.PE(file_path, fast_load=True)

        features['sections_count'] = len(pe.sections) if hasattr(pe, 'sections') else 0
        missing_flags['sections_count_missing'] = 0 if hasattr(pe, 'sections') else 1

        features['symbols_count'] = len(pe.SYMBOL_TABLE) if hasattr(pe, 'SYMBOL_TABLE') else 0
        missing_flags['symbols_count_missing'] = 0 if hasattr(pe, 'SYMBOL_TABLE') else 1

        features['imports_count'] = 0
        features['exports_count'] = 0

        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            imports = []
            api_names = []
            dll_names = []

            features['imports_count'] = len(pe.DIRECTORY_ENTRY_IMPORT)
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8').lower() if entry.dll else ''
                dll_names.append(dll_name)

                for imp in entry.imports:
                    if imp.name:
                        func_name = imp.name.decode('utf-8') if imp.name else ''
                        imports.append((dll_name, func_name))
                        api_names.append(func_name)

            features['unique_imports'] = len(set(imports))
            features['unique_dlls'] = len(set(dll_names))
            features['unique_apis'] = len(set(api_names))

            if dll_names:
                dll_name_lengths = [len(name) for name in dll_names if name]
                features['dll_name_avg_length'] = np.mean(dll_name_lengths)
                features['dll_name_max_length'] = np.max(dll_name_lengths)
                features['dll_name_min_length'] = np.min(dll_name_lengths)
                missing_flags['dll_stats_missing'] = 0
            else:
                missing_flags['dll_stats_missing'] = 1

            system_dlls = {'kernel32', 'user32', 'gdi32', 'advapi32', 'shell32', 'ole32', 'comctl32'}
            imported_system_dlls = set(dll.split('.')[0].lower() for dll in dll_names if dll) & system_dlls
            features['imported_system_dlls_count'] = len(imported_system_dlls)
            missing_flags['imported_system_dlls_missing'] = 0
        else:
            features['unique_imports'] = 0
            features['unique_dlls'] = 0
            features['unique_apis'] = 0
            features['dll_name_avg_length'] = 0
            features['dll_name_max_length'] = 0
            features['dll_name_min_length'] = 0
            features['imported_system_dlls_count'] = 0
            missing_flags['dll_stats_missing'] = 1
            missing_flags['imported_system_dlls_missing'] = 1

        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            features['exports_count'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)

            export_names = []
            for symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if symbol.name:
                    export_names.append(symbol.name.decode('utf-8'))

            if export_names:
                export_name_lengths = [len(name) for name in export_names]
                features['export_name_avg_length'] = np.mean(export_name_lengths)
                features['export_name_max_length'] = np.max(export_name_lengths)
                features['export_name_min_length'] = np.min(export_name_lengths)

                features['exports_density'] = len(export_names) / (features['size'] + 1)
                missing_flags['export_stats_missing'] = 0
            else:
                features['export_name_avg_length'] = 0
                features['export_name_max_length'] = 0
                features['export_name_min_length'] = 0
                features['exports_density'] = 0
                missing_flags['export_stats_missing'] = 1
        else:
            features['exports_count'] = 0
            features['export_name_avg_length'] = 0
            features['export_name_max_length'] = 0
            features['export_name_min_length'] = 0
            features['exports_density'] = 0
            missing_flags['export_stats_missing'] = 1

        if hasattr(pe, 'sections'):
            section_names = []
            section_sizes = []
            section_vsizes = []
            section_chars = []
            code_section_size = 0
            data_section_size = 0
            code_section_vsize = 0
            data_section_vsize = 0

            executable_sections_count = 0
            writable_sections_count = 0
            readable_sections_count = 0
            non_standard_executable_sections_count = 0
            rwx_sections_count = 0

            common_executable_section_names = {'.text', 'text', '.code'}

            for section in pe.sections:
                try:
                    name = section.Name.decode('utf-8').strip('\x00')
                    section_names.append(name)
                    section_sizes.append(section.SizeOfRawData)
                    section_vsizes.append(section.VirtualSize)
                    section_chars.append(section.Characteristics)

                    if section.Characteristics & 0x20000000:
                        executable_sections_count += 1
                        code_section_size += section.SizeOfRawData
                        code_section_vsize += section.VirtualSize

                        if name.lower() not in common_executable_section_names:
                            non_standard_executable_sections_count += 1
                    if section.Characteristics & 0x80000000:
                        writable_sections_count += 1
                    if section.Characteristics & 0x40000000:
                        readable_sections_count += 1
                        data_section_size += section.SizeOfRawData
                        data_section_vsize += section.VirtualSize

                    if (section.Characteristics & 0x20000000) and (section.Characteristics & 0x80000000):
                        features['executable_writable_sections'] = features.get('executable_writable_sections', 0) + 1
                        rwx_sections_count += 1
                except Exception:
                    increment_error()
                    pass

            features['section_names_count'] = len(section_names)
            features['section_total_size'] = sum(section_sizes)
            features['section_total_vsize'] = sum(section_vsizes)
            features['avg_section_size'] = np.mean(section_sizes) if section_sizes else 0
            features['avg_section_vsize'] = np.mean(section_vsizes) if section_vsizes else 0
            features['max_section_size'] = np.max(section_sizes) if section_sizes else 0
            features['min_section_size'] = np.min(section_sizes) if section_sizes else 0
            features['code_section_ratio'] = code_section_size / (features['section_total_size'] + 1)
            features['data_section_ratio'] = data_section_size / (features['section_total_size'] + 1)
            features['code_vsize_ratio'] = code_section_vsize / (features['section_total_vsize'] + 1)
            features['data_vsize_ratio'] = data_section_vsize / (features['section_total_vsize'] + 1)

            features['executable_sections_count'] = executable_sections_count
            features['writable_sections_count'] = writable_sections_count
            features['readable_sections_count'] = readable_sections_count
            features['executable_sections_ratio'] = executable_sections_count / (len(section_names) + 1)
            features['writable_sections_ratio'] = writable_sections_count / (len(section_names) + 1)
            features['readable_sections_ratio'] = readable_sections_count / (len(section_names) + 1)
            features['non_standard_executable_sections_count'] = non_standard_executable_sections_count
            features['rwx_sections_count'] = rwx_sections_count
            features['rwx_sections_ratio'] = rwx_sections_count / (len(section_names) + 1)

            if features['section_total_size'] > 0:
                features['executable_code_density'] = code_section_size / features['section_total_size']
            else:
                features['executable_code_density'] = 0

            if section_sizes:
                features['section_size_std'] = np.std(section_sizes)
                features['section_size_cv'] = np.std(section_sizes) / (np.mean(section_sizes) + 1e-8)
            else:
                features['section_size_std'] = 0
                features['section_size_cv'] = 0

            if section_names:
                section_name_lengths = [len(name) for name in section_names]
                features['section_name_avg_length'] = np.mean(section_name_lengths)
                features['section_name_max_length'] = np.max(section_name_lengths)
                features['section_name_min_length'] = np.min(section_name_lengths)
                missing_flags['section_name_stats_missing'] = 0
            else:
                features['section_name_avg_length'] = 0
                features['section_name_max_length'] = 0
                features['section_name_min_length'] = 0
                missing_flags['section_name_stats_missing'] = 1

            special_char_count = 0
            total_chars = 0
            for name in section_names:
                total_chars += len(name)
                for c in name:
                    if not (c.isalnum() or c in '_.'):
                        special_char_count += 1

            features['special_char_ratio'] = special_char_count / (total_chars + 1)

            long_sections = [name for name in section_names if len(name) > 6]
            short_sections = [name for name in section_names if len(name) < 3]
            features['long_sections_count'] = len(long_sections)
            features['short_sections_count'] = len(short_sections)
            features['long_sections_ratio'] = len(long_sections) / (len(section_names) + 1)
            features['short_sections_ratio'] = len(short_sections) / (len(section_names) + 1)
            missing_flags['sections_details_missing'] = 0
        else:
            features['section_name_avg_length'] = 0
            features['section_name_max_length'] = 0
            features['section_name_min_length'] = 0
            features['max_section_size'] = 0
            features['min_section_size'] = 0
            features['code_section_ratio'] = 0
            features['data_section_ratio'] = 0
            features['code_vsize_ratio'] = 0
            features['data_vsize_ratio'] = 0
            features['long_sections_count'] = 0
            features['short_sections_count'] = 0
            features['section_size_std'] = 0
            features['section_size_cv'] = 0
            features['executable_writable_sections'] = 0
            features['non_standard_executable_sections_count'] = 0
            features['rwx_sections_count'] = 0
            features['rwx_sections_ratio'] = 0.0
            missing_flags['section_name_stats_missing'] = 1
            missing_flags['sections_details_missing'] = 1

            common_sections = ['.text', '.data', '.rdata', '.reloc', '.rsrc']
            for sec in common_sections:
                features[f'has_{sec}_section'] = 0

        if hasattr(pe.OPTIONAL_HEADER, 'Subsystem'):
            features['subsystem'] = pe.OPTIONAL_HEADER.Subsystem
            missing_flags['subsystem_missing'] = 0
        else:
            features['subsystem'] = 0
            missing_flags['subsystem_missing'] = 1

        if hasattr(pe.OPTIONAL_HEADER, 'DllCharacteristics'):
            features['dll_characteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics

            features['has_nx_compat'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x100 else 0
            features['has_aslr'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x40 else 0
            features['has_seh'] = 1 if not (pe.OPTIONAL_HEADER.DllCharacteristics & 0x400) else 0
            features['has_guard_cf'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x4000 else 0
            missing_flags['dll_characteristics_missing'] = 0
        else:
            features['dll_characteristics'] = 0
            features['has_nx_compat'] = 0
            features['has_aslr'] = 0
            features['has_seh'] = 0
            features['has_guard_cf'] = 0
            missing_flags['dll_characteristics_missing'] = 1

        features['has_resources'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') else 0
        features['has_debug_info'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') else 0
        features['has_tls'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0
        features['has_relocs'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0
        features['has_exceptions'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION') else 0
        missing_flags['directory_entries_missing'] = 0 if (hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') or
                                                          hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') or
                                                          hasattr(pe, 'DIRECTORY_ENTRY_TLS') or
                                                          hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') or
                                                          hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION')) else 1

        try:
            with open(file_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                pe_end_offset = pe.sections[-1].PointerToRawData + pe.sections[-1].SizeOfRawData if hasattr(pe, 'sections') and pe.sections else file_size

                trailing_data_size = file_size - pe_end_offset
                features['trailing_data_size'] = trailing_data_size
                features['trailing_data_ratio'] = trailing_data_size / (file_size + 1)

                features['has_large_trailing_data'] = 1 if trailing_data_size > 1024 * 1024 else 0
                missing_flags['trailing_data_missing'] = 0
        except Exception:
            increment_error()
            features['trailing_data_size'] = 0
            features['trailing_data_ratio'] = 0
            features['has_large_trailing_data'] = 0
            missing_flags['trailing_data_missing'] = 1

        try:
            pe_header_size = pe.OPTIONAL_HEADER.SizeOfHeaders
            features['pe_header_size'] = pe_header_size
            features['header_size_ratio'] = pe_header_size / (features['size'] + 1)
            missing_flags['header_info_missing'] = 0
        except Exception:
            increment_error()
            features['pe_header_size'] = 0
            features['header_size_ratio'] = 0
            missing_flags['header_info_missing'] = 1

    except Exception as e:
        increment_error()
        default_keys = [
            'sections_count', 'symbols_count',
            'imports_count', 'exports_count', 'unique_imports', 'unique_dlls',
            'unique_apis', 'section_names_count', 'section_total_size',
            'section_total_vsize', 'avg_section_size', 'avg_section_vsize',
            'subsystem', 'dll_characteristics', 'code_section_ratio',
            'data_section_ratio', 'code_vsize_ratio', 'data_vsize_ratio',
            'has_nx_compat', 'has_aslr', 'has_seh', 'has_guard_cf', 'has_resources',
            'has_debug_info', 'has_tls', 'has_relocs', 'has_exceptions',
            'dll_name_avg_length', 'dll_name_max_length', 'dll_name_min_length',
            'section_name_avg_length', 'section_name_max_length', 'section_name_min_length',
            'export_name_avg_length', 'export_name_max_length', 'export_name_min_length',
            'max_section_size', 'min_section_size', 'entry_point_ratio',
            'long_sections_count', 'short_sections_count',
            'section_size_std', 'section_size_cv', 'executable_writable_sections',
            'file_entropy_avg', 'file_entropy_min', 'file_entropy_max', 'file_entropy_range',
            'zero_byte_ratio', 'printable_byte_ratio', 'trailing_data_size', 'trailing_data_ratio',
            'imported_system_dlls_count', 'exports_density',
            'has_large_trailing_data', 'pe_header_size', 'header_size_ratio',
            'file_entropy_std', 'file_entropy_q25', 'file_entropy_q75',
            'file_entropy_median', 'high_entropy_ratio', 'low_entropy_ratio',
            'entropy_change_rate', 'entropy_change_std',
            'executable_sections_count', 'writable_sections_count', 'readable_sections_count',
            'executable_sections_ratio', 'writable_sections_ratio', 'readable_sections_ratio',
            'executable_code_density',
            'non_standard_executable_sections_count', 'rwx_sections_count', 'rwx_sections_ratio',
            'special_char_ratio', 'long_sections_ratio', 'short_sections_ratio'
        ]

        for key in default_keys:
            features[key] = 0

        common_sections = ['.text', '.data', '.rdata', '.reloc', '.rsrc']
        for sec in common_sections:
            features[f'has_{sec}_section'] = 0

        missing_flag_names = ['sections_count_missing', 'symbols_count_missing', 'dll_stats_missing',
                             'imported_system_dlls_missing', 'export_stats_missing', 'section_name_stats_missing',
                             'sections_details_missing', 'subsystem_missing', 'dll_characteristics_missing',
                             'directory_entries_missing', 'trailing_data_missing', 'header_info_missing']
        for flag in missing_flag_names:
            missing_flags[flag] = 1

    features.update(missing_flags)
    return features

def extract_lightweight_pe_features(file_path):

    feature_vector = np.zeros(256, dtype=np.float32)
    try:
        pe = pefile.PE(file_path, fast_load=True)

        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                if entry.dll:
                    dll_name = entry.dll.decode('utf-8').lower()
                    dll_hash = int(hashlib.sha256(dll_name.encode('utf-8')).hexdigest(), 16)
                    feature_vector[dll_hash % 128] = 1

            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        api_name = imp.name.decode('utf-8')
                        api_hash = int(hashlib.sha256(api_name.encode('utf-8')).hexdigest(), 16)
                        feature_vector[128 + (api_hash % 128)] = 1

        if hasattr(pe, 'sections'):
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
                section_hash = int(hashlib.sha256(section_name.encode('utf-8')).hexdigest(), 16)
                feature_vector[section_hash % 32 + 224] = 1

        norm = np.linalg.norm(feature_vector)
        if norm > 0 and not np.isnan(norm):
            feature_vector /= norm

        return feature_vector

    except Exception:
        increment_error()
        return feature_vector

def extract_combined_pe_features(file_path):
    lightweight_features = extract_lightweight_pe_features(file_path)

    enhanced_features = extract_enhanced_pe_features(file_path)
    file_attrs = extract_file_attributes(file_path)

    all_features = {}
    all_features.update(enhanced_features)
    all_features.update(file_attrs)

    combined_vector = np.zeros(1000, dtype=np.float32)

    combined_vector[:256] = lightweight_features * 1.5

    max_file_size = 100 * 1024 * 1024
    max_timestamp = 2147483647

    feature_order = [
        'size', 'log_size', 'sections_count', 'symbols_count', 'imports_count', 'exports_count',
        'unique_imports', 'unique_dlls', 'unique_apis', 'section_names_count', 'section_total_size',
        'section_total_vsize', 'avg_section_size', 'avg_section_vsize', 'subsystem', 'dll_characteristics',
        'code_section_ratio', 'data_section_ratio', 'code_vsize_ratio', 'data_vsize_ratio',
        'has_nx_compat', 'has_aslr', 'has_seh', 'has_guard_cf', 'has_resources', 'has_debug_info',
        'has_tls', 'has_relocs', 'has_exceptions', 'dll_name_avg_length', 'dll_name_max_length',
        'dll_name_min_length', 'section_name_avg_length', 'section_name_max_length', 'section_name_min_length',
        'export_name_avg_length', 'export_name_max_length', 'export_name_min_length', 'max_section_size',
        'min_section_size', 'long_sections_count', 'short_sections_count', 'section_size_std', 'section_size_cv',
        'executable_writable_sections', 'file_entropy_avg', 'file_entropy_min', 'file_entropy_max',
        'file_entropy_range', 'zero_byte_ratio', 'printable_byte_ratio', 'trailing_data_size',
        'trailing_data_ratio', 'imported_system_dlls_count', 'exports_density', 'has_large_trailing_data',
        'pe_header_size', 'header_size_ratio', 'file_entropy_std', 'file_entropy_q25', 'file_entropy_q75',
        'file_entropy_median', 'high_entropy_ratio', 'low_entropy_ratio', 'entropy_change_rate',
        'entropy_change_std', 'executable_sections_count', 'writable_sections_count', 'readable_sections_count',
        'executable_sections_ratio', 'writable_sections_ratio', 'readable_sections_ratio', 'executable_code_density',
        'non_standard_executable_sections_count', 'rwx_sections_count', 'rwx_sections_ratio',
        'special_char_ratio', 'long_sections_ratio', 'short_sections_ratio',

        'size_missing', 'log_size_missing', 'file_entropy_missing', 'file_entropy_percentiles_missing',
        'byte_stats_missing', 'sections_count_missing', 'symbols_count_missing', 'dll_stats_missing',
        'imported_system_dlls_missing', 'export_stats_missing', 'section_name_stats_missing',
        'sections_details_missing', 'subsystem_missing', 'dll_characteristics_missing',
        'directory_entries_missing', 'trailing_data_missing', 'header_info_missing'
    ]

    common_sections = ['.text', '.data', '.rdata', '.reloc', '.rsrc']
    for sec in common_sections:
        feature_order.append(f'has_{sec}_section')

    for i, key in enumerate(feature_order):
        if 256 + i >= 1000:
            break

        if key in all_features:
            val = all_features[key]
            if 'size' in key and isinstance(val, (int, float)):
                val = val / max_file_size
            elif key == 'timestamp' and isinstance(val, (int, float)):
                val = val / max_timestamp
            elif key == 'timestamp_year' and isinstance(val, (int, float)):
                val = (val - 1970) / (2038 - 1970)
            elif key.startswith('has_') and isinstance(val, (int, float)):
                val = float(val)
            elif key == 'log_size' and isinstance(val, (int, float)):
                val = val / np.log(max_file_size)

            combined_vector[256 + i] = val * 0.8 if isinstance(val, (int, float)) else 0

    norm = np.linalg.norm(combined_vector)
    if norm > 0 and not np.isnan(norm):
        combined_vector /= norm
    else:
        combined_vector = np.zeros(1000, dtype=np.float32)

    return combined_vector

def process_file_worker(args):
    file_path, label, output_dir = args
    before = ERROR_COUNT

    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        increment_error()
        return {'filename': 'unknown', 'status': 'failed', 'error': f'Could not read file: {file_path}', 'errors': ERROR_COUNT - before}

    filename = file_hash
    output_npz_path = os.path.join(output_dir, filename + '.npz')

    if os.path.exists(output_npz_path):
        return {'filename': filename, 'status': 'skipped', 'label': label, 'errors': ERROR_COUNT - before}

    byte_sequence = extract_byte_sequence(file_path)
    if byte_sequence is None:
        return {'filename': filename, 'status': 'failed', 'error': 'Could not read byte sequence.', 'errors': ERROR_COUNT - before}

    pe_features = extract_combined_pe_features(file_path)

    try:
        np.savez_compressed(
            output_npz_path,
            byte_sequence=byte_sequence,
            pe_features=pe_features
        )
        return {'filename': filename, 'status': 'success', 'label': label, 'errors': ERROR_COUNT - before}
    except Exception as e:
        increment_error()
        return {'filename': filename, 'status': 'failed', 'error': str(e), 'errors': ERROR_COUNT - before}

def extract_features_in_memory(input_file_path, max_file_size=256*1024):

    global MAX_FILE_SIZE
    original_max_size = MAX_FILE_SIZE
    MAX_FILE_SIZE = max_file_size

    try:
        byte_sequence = extract_byte_sequence(input_file_path)
        if byte_sequence is None:
            raise Exception("Failed to extract byte sequence")

        pe_features = extract_combined_pe_features(input_file_path)

        return byte_sequence, pe_features
    except Exception as e:
        increment_error()
        print(f"[!] Failed to extract in-memory features for file {input_file_path}: {e}")
        return None, None
    finally:
        MAX_FILE_SIZE = original_max_size

def process_file_directory(input_file_path, output_file_path, max_file_size=256*1024):

    byte_sequence, pe_features = extract_features_in_memory(input_file_path, max_file_size)
    if byte_sequence is None or pe_features is None:
        raise Exception(f"Failed to process file {input_file_path}")

    np.savez_compressed(
        output_file_path,
        byte_sequence=byte_sequence,
        pe_features=pe_features
    )

    print(f"[+] Successfully processed file: {input_file_path} -> {output_file_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    from tqdm import tqdm

    BENIGN_DIR = os.path.join(base_dir, 'benign_samples')
    MALICIOUS_DIR = os.path.join(base_dir, 'malicious_samples')

    PROCESSED_DATA_DIR = os.path.join(base_dir, 'data', 'processed_lightgbm')
    METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'metadata.json')

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    def collect_tasks_recursive(base_directory, output_dir):
        tasks = []
        if not os.path.isdir(base_directory): return tasks
        label = 'benign' if 'benign_samples' in base_directory else 'malicious'
        for root, _, files in os.walk(base_directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                tasks.append((file_path, label, output_dir))
        return tasks

    benign_tasks = collect_tasks_recursive(BENIGN_DIR, PROCESSED_DATA_DIR)
    malicious_tasks = collect_tasks_recursive(MALICIOUS_DIR, PROCESSED_DATA_DIR)
    all_tasks = benign_tasks + malicious_tasks

    if not all_tasks:
        print("\n[!] No sample files found.")
        sys.exit()

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_file_worker, all_tasks), total=len(all_tasks)))

    file_to_label = {}

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            file_to_label = json.load(f)

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for r in results:
        status = r.get('status')
        if status == 'success':
            success_count += 1
            file_to_label[r['filename'] + '.npz'] = r['label']
        elif status == 'skipped':
            skipped_count += 1
            if r['filename'] + '.npz' not in file_to_label:
                 file_to_label[r['filename'] + '.npz'] = r['label']
        else:
            failed_count += 1

    with open(METADATA_FILE, 'w') as f:
        json.dump(file_to_label, f, indent=4)

    total_errors = sum(r.get('errors', 0) for r in results)
    print(f"\n[!] 本次运行共捕获到 {total_errors} 个错误")
