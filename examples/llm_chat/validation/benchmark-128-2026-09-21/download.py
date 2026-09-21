from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
models = [
    ('qwen3-0.6b', 'Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca'),
    ('llama3-8b', 'NousResearch/Meta-Llama-3-8B-Instruct', '53346005fb0ef11d3b6a83b12c895cca40156b6c'),
    ('gemma3-4b', 'unsloth/gemma-3-4b-it', 'bf46152c47f5dd20b896357cb51abc4c03b8ee8c'),
    ('qwen3-4b', 'Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
]
def download(item):
    name, repo, revision = item
    revision = revision or HfApi().model_info(repo).sha
    path = Path('/dev/shm/llm-chat-bench-checkpoints') / name
    print(f'Downloading {repo}@{revision} to {path}', flush=True)
    snapshot_download(repo, revision=revision, local_dir=path, allow_patterns=['*.json', '*.safetensors', '*.jinja'], max_workers=4)
    (path / 'REVISION').write_text(revision + '\n')
    (path / 'BENCH_SOURCE.json').write_text(json.dumps({'repo': repo, 'revision': revision}))
    print(f'READY {name}', flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(download, models))
