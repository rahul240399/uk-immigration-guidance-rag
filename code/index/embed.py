"""
Dense embeddings for retrieval indexes.

Usage::

    python -m code.index.embed --config config/paths.yaml --chunker para \\
        --model BAAI/bge-base-en-v1.5 [--batch-size 64]
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from code.common.run_registry import load_paths, start_run

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _best_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_chunks(index_dir: Path, chunker: str) -> tuple[list[str], list[str]]:
    """Load chunk texts and ids. Returns (ids, texts)."""
    if chunker == "win64":
        fname = "windows.jsonl"
        id_key = "window_id"
    else:
        fname = "chunks.jsonl"
        id_key = "chunk_id"

    ids, texts = [], []
    with open(index_dir / fname, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            ids.append(c[id_key])
            texts.append(c["text"])
    return ids, texts


def encode_query(model: SentenceTransformer, text: str) -> np.ndarray:
    """Encode a query with the bge v1.5 instruction prefix, normalized."""
    return model.encode(QUERY_PREFIX + text, normalize_embeddings=True)


def main():
    ap = argparse.ArgumentParser(description="Build dense embeddings for an index")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--chunker", required=True, choices=["para", "parentchild", "win64"])
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    paths = load_paths(args.config)
    idx_root = Path(paths["index"])
    idx_dir = idx_root / args.chunker

    if not idx_dir.exists():
        log.error("Index dir %s does not exist; run build_index first", idx_dir)
        exit(1)

    device = _best_device()
    log.info("Loading model %s on %s", args.model, device)
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = 512

    # Get model revision from the config file in the cache
    model_revision = "unknown"
    try:
        cfg_path = Path(model.model_card_data.model_id) if hasattr(model, 'model_card_data') else None
        # Try reading from the downloaded model's config
        import huggingface_hub
        info = huggingface_hub.model_info(args.model)
        model_revision = info.sha
    except Exception:
        pass

    ids, texts = load_chunks(idx_dir, args.chunker)
    n = len(ids)
    log.info("Loaded %d chunks from %s", n, idx_dir)

    # Count truncated (> 512 tokens)
    tokenizer = model.tokenizer
    truncated = 0
    for t in texts:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if len(toks) > 512:
            truncated += 1

    # Encode
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=args.batch_size,
                              normalize_embeddings=True, show_progress_bar=True)
    wall_time = time.time() - t0

    emb = np.array(embeddings, dtype=np.float32)
    assert emb.shape == (n, 768), f"Unexpected shape: {emb.shape}"

    # Write .npy and id_map
    model_slug = args.model.replace("/", "_")
    npy_path = idx_dir / f"emb_{model_slug}.npy"
    np.save(npy_path, emb)

    id_map_path = idx_dir / f"emb_{model_slug}.id_map.json"
    id_map_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")

    npy_sha = _sha256_file(npy_path)
    id_sha = _sha256_file(id_map_path)

    # Register run
    ctx = start_run("embed", f"{args.chunker}-bge-base", {
        "chunker": args.chunker,
        "model": args.model,
        "model_revision": model_revision,
        "device": device,
        "batch_size": args.batch_size,
        "max_seq_length": 512,
        "query_prefix": QUERY_PREFIX,
        "chunk_count": n,
        "truncated": truncated,
        "wall_time_s": round(wall_time, 2),
        "npy_sha256": npy_sha,
        "id_map_sha256": id_sha,
    }, paths)
    ctx.log(f"model={args.model} rev={model_revision} device={device}")
    ctx.log(f"chunks={n} truncated={truncated} time={wall_time:.1f}s")
    ctx.log(f"npy: {npy_path} sha256={npy_sha}")
    ctx.log(f"id_map: {id_map_path} sha256={id_sha}")

    # Write embed.log in index dir
    embed_log = idx_dir / "embed.log"
    with open(embed_log, "w") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"model_revision: {model_revision}\n")
        f.write(f"device: {device}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"max_seq_length: 512\n")
        f.write(f"chunk_count: {n}\n")
        f.write(f"truncated: {truncated}\n")
        f.write(f"wall_time_s: {wall_time:.2f}\n")
        f.write(f"npy_sha256: {npy_sha}\n")
        f.write(f"id_map_sha256: {id_sha}\n")

    ctx.finish("ok", f"{args.chunker}: {n} chunks, {device}, {wall_time:.1f}s")

    print(f"Chunker: {args.chunker}")
    print(f"N: {n}")
    print(f"Truncated: {truncated}")
    print(f"Device: {device}")
    print(f"Time: {wall_time:.1f}s")
    print(f"npy sha256: {npy_sha}")
    print(f"id_map sha256: {id_sha}")


if __name__ == "__main__":
    main()
