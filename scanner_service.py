import os
import shutil
import tempfile
import sys
from pathlib import Path
from threading import Lock
from typing import Optional, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from scanner import MalwareScanner

BASE_DIR = getattr(sys, '_MEIPASS', os.path.abspath('.'))
DEFAULT_LIGHTGBM_MODEL = os.path.join(BASE_DIR, 'saved_models', 'lightgbm_model.txt')
DEFAULT_CLUSTER_MAPPING = os.path.join(BASE_DIR, 'hdbscan_cluster_results', 'file_cluster_mapping.json')
DEFAULT_FAMILY_MAPPING = os.path.join(BASE_DIR, 'hdbscan_cluster_results', 'family_names_mapping.json')
DEFAULT_CACHE_PATH = os.path.join(BASE_DIR, 'scan_cache.json')
DEFAULT_MAX_FILE_SIZE = 256 * 1024


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


class FileScanRequest(BaseModel):
    file_path: str = Field(..., description='需要扫描的文件绝对路径')


app = FastAPI(title='KoloVirusDetector Scanner Service', version='1.0.0')

_scanner_lock = Lock()
_scanner_instance: Optional[MalwareScanner] = None


def _prefer_gz(path: str) -> str:
    gz = path + ('.gz' if not path.endswith('.gz') else '')
    return gz if os.path.exists(gz) and not os.path.exists(path) else path

def _build_scanner() -> MalwareScanner:
    lightgbm_model_path = _prefer_gz(os.getenv('SCANNER_LIGHTGBM_MODEL_PATH', DEFAULT_LIGHTGBM_MODEL))
    cluster_mapping_path = _prefer_gz(os.getenv('SCANNER_CLUSTER_MAPPING_PATH', DEFAULT_CLUSTER_MAPPING))
    family_names_path = _prefer_gz(os.getenv('SCANNER_FAMILY_NAMES_PATH', DEFAULT_FAMILY_MAPPING))
    cache_file = os.getenv('SCANNER_CACHE_PATH', DEFAULT_CACHE_PATH)
    max_file_size = _env_int('SCANNER_MAX_FILE_SIZE', DEFAULT_MAX_FILE_SIZE)

    missing_paths: List[str] = [
        p for p in [lightgbm_model_path, cluster_mapping_path, family_names_path]
        if not os.path.exists(p)
    ]
    if missing_paths:
        raise RuntimeError(f'以下必需文件不存在: {missing_paths}')

    return MalwareScanner(
        lightgbm_model_path=lightgbm_model_path,
        cluster_mapping_path=cluster_mapping_path,
        family_names_path=family_names_path,
        max_file_size=max_file_size,
        cache_file=None,
        enable_cache=False,
    )


def get_scanner() -> MalwareScanner:
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = _build_scanner()
    return _scanner_instance


@app.on_event('startup')
def _startup() -> None:
    get_scanner()


@app.get('/health')
def health() -> dict:
    scanner = get_scanner()
    return {
        'status': 'ok',
        'cache_size': len(scanner.scan_cache),
        'model_path': os.getenv('SCANNER_LIGHTGBM_MODEL_PATH', DEFAULT_LIGHTGBM_MODEL)
    }


@app.post('/scan/file')
def scan_file(request: FileScanRequest) -> dict:
    scanner = get_scanner()
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail='文件不存在')

    with _scanner_lock:
        result = scanner.scan_file(request.file_path)

    if result is None:
        raise HTTPException(status_code=400, detail='文件不是有效的PE或扫描失败')
    return result


@app.post('/scan/upload')
async def scan_upload(file: UploadFile = File(...)) -> dict:
    scanner = get_scanner()
    suffix = Path(file.filename or '').suffix or '.bin'

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_path = tmp_file.name
    finally:
        file.file.close()

    try:
        with _scanner_lock:
            result = scanner.scan_file(temp_path)
    finally:
        os.unlink(temp_path)

    if result is None:
        raise HTTPException(status_code=400, detail='文件不是有效的PE或扫描失败')

    result['original_filename'] = file.filename
    return result


@app.post('/cache/save')
def flush_cache() -> dict:
    scanner = get_scanner()

    if not getattr(scanner, 'enable_cache', False):
        return {'status': 'disabled', 'cache_size': 0}

    with _scanner_lock:
        scanner._save_cache()

    return {'status': 'saved', 'cache_size': len(scanner.scan_cache)}


if __name__ == '__main__':
    import uvicorn

    port = _env_int('SCANNER_SERVICE_PORT', 8000)
    uvicorn.run(app, host='0.0.0.0', port=port, reload=False)

