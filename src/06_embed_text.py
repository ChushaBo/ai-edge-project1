"""Embed claim narratives (raw and damage-figure-masked) and reduce with PCA.

Method (decided 2026-11-15, docs/decisions.md "Text embedding model"):
  1. A claim's text is a set of pieces: the episode narratives and event narratives of
     its matched NOAA events (from 05), each unique piece once.
  2. Each piece is split into chunks of <= embedding.chunk_words words (the model reads
     at most 256 tokens and would silently drop the rest), each chunk is embedded, and
     the piece vector is the mean of its chunk vectors.
  3. The claim vector is the mean of its piece vectors.
  4. A masked variant repeats 1-3 after replacing dollar amounts and counts of damaged
     things with placeholders (utils.text.mask_damage_numbers), to test whether the
     text's value comes from storm description or from leaked damage figures.
  5. PCA to pca.n_components, fit on TRAIN claims with a narrative only, applied to all
     claims. Claims without a narrative get zeros in every component.
Embeddings are cached per (model, chunk text), so duplicates are embedded once and
reruns are fast. Quarantined claims (config quarantine) are skipped.

Inputs:
  - data/interim/claims_with_text.parquet (from 05)
  - data/raw/storms_fl.parquet (episode and event narratives, by ID)
  - sentence-transformers model embedding.model (downloaded on first run)
  - config.yaml: embedding.*, pca.n_components, random_seed, quarantine.*, paths.*
Outputs:
  - data/processed/text_features.parquet: id, split, has_narrative, txt_pc01..NN
    (raw text), txtm_pc01..NN (masked text).
  - data/processed/embedding_cache.parquet: chunk embeddings by cache key.
  - data/models/pca_text.joblib, data/models/pca_text_masked.joblib: fitted PCA, plus
    the ids of the claims each was fit on (for the train-only test).
  - outputs/tables/text_pca_variance.csv: explained variance per component, both variants.
"""

import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.config import get_path, load_config, require
from utils.io import write_parquet_atomic
from utils.log import get_logger, log_rows
from utils.matching import exclude_quarantined
from utils.text import chunk_words, mask_damage_numbers, text_key

logger = get_logger("06_embed_text")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def claim_pieces(claims: pd.DataFrame, storms: pd.DataFrame) -> pd.Series:
    """Per claim id: tuple of unique, non-empty narrative pieces (episodes, then events)."""
    def as_dict(keys: pd.Series, texts: pd.Series) -> dict[int, str]:
        # missing narratives (NA) become empty strings and are skipped below
        return {int(k): ("" if pd.isna(t) else str(t).strip()) for k, t in zip(keys, texts)}

    ep = storms.drop_duplicates("EPISODE_ID")
    episodes = as_dict(ep["EPISODE_ID"], ep["EPISODE_NARRATIVE"])
    events = as_dict(storms["EVENT_ID"], storms["EVENT_NARRATIVE"])

    def lookup(ids: str, table: dict[int, str]) -> list[str]:
        return [table.get(int(i), "") for i in ids.split(";") if i]

    pieces = {}
    for cid, ep_ids, ev_ids in zip(claims["id"], claims["matched_episode_ids"],
                                   claims["matched_event_ids"]):
        seen, out = set(), []
        for text in lookup(ep_ids, episodes) + lookup(ev_ids, events):
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        pieces[cid] = tuple(out)
    return pd.Series(pieces)


class Embedder:
    """Chunk-level embedding with a persistent cache keyed by (model, chunk text)."""

    def __init__(self, model_name: str, cache_path, batch_size: int, chunk_size: int):
        self.model_name, self.cache_path = model_name, cache_path
        self.batch_size, self.chunk_size = batch_size, chunk_size
        self.cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self.cache = dict(zip(df["key"], df["vector"].map(np.asarray)))
        self.n_cached_at_start = len(self.cache)
        self._model = None

    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("Loaded %s (max_seq_length %d)", self.model_name, self._model.max_seq_length)
        return self._model

    def piece_vectors(self, pieces: list[str]) -> dict[str, np.ndarray]:
        """Mean chunk vector for each unique piece of text."""
        chunks = {p: chunk_words(p, self.chunk_size) for p in pieces}
        todo = sorted({c for cs in chunks.values() for c in cs
                       if text_key(self.model_name, c) not in self.cache})
        if todo:
            logger.info("Embedding %d new chunks (%d already cached)",
                        len(todo), len({c for cs in chunks.values() for c in cs}) - len(todo))
            vecs = self.model().encode(todo, batch_size=self.batch_size, show_progress_bar=True,
                                       normalize_embeddings=True, convert_to_numpy=True)
            self.cache.update({text_key(self.model_name, c): v for c, v in zip(todo, vecs)})
        return {p: np.mean([self.cache[text_key(self.model_name, c)] for c in cs], axis=0)
                for p, cs in chunks.items() if cs}

    def save(self) -> None:
        df = pd.DataFrame({"key": list(self.cache), "vector": [v.tolist() for v in self.cache.values()]})
        write_parquet_atomic(df, self.cache_path)


def claim_matrix(pieces: pd.Series, vectors: dict[str, np.ndarray], dim: int) -> np.ndarray:
    """Claim vector = mean of its piece vectors (zeros when a claim has no pieces).
    Claims with the same set of pieces share one computation."""
    by_set: dict[tuple, np.ndarray] = {}
    out = np.zeros((len(pieces), dim), dtype=np.float32)
    for i, ps in enumerate(pieces.to_numpy()):
        if not ps:
            continue
        if ps not in by_set:
            by_set[ps] = np.mean([vectors[p] for p in ps if p in vectors], axis=0)
        out[i] = by_set[ps]
    return out


def fit_pca_train_only(X: np.ndarray, fit_mask: np.ndarray, n_components: int, seed: int) -> PCA:
    """Fit PCA on the rows selected by fit_mask only (train claims with a narrative)."""
    if fit_mask.sum() < n_components:
        raise SystemExit(f"STOP: only {int(fit_mask.sum())} rows to fit {n_components} PCA components")
    return PCA(n_components=n_components, random_state=seed).fit(X[fit_mask])


def pca_features(pca: PCA, X: np.ndarray, has_text: np.ndarray, prefix: str) -> pd.DataFrame:
    Z = pca.transform(X)
    Z[~has_text] = 0.0  # claims without a narrative: zeros (has_narrative flags them)
    cols = [f"{prefix}_pc{i + 1:02d}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z.astype(np.float32), columns=cols)


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    seed = int(require(cfg, "random_seed"))
    set_seeds(seed)
    k = int(require(cfg, "pca.n_components"))
    model_name = require(cfg, "embedding.model")
    interim, processed, models = (get_path(cfg, "interim"), get_path(cfg, "processed"),
                                  get_path(cfg, "models"))

    claims = pd.read_parquet(interim / "claims_with_text.parquet",
                             columns=["id", "split", "has_narrative", "quarantined",
                                      "matched_episode_ids", "matched_event_ids"])
    log_rows(logger, claims, "claims_with_text")
    # Text features only ever feed M2 / M2_masked (both share the same quarantine scope).
    claims = exclude_quarantined(claims, cfg, model="M2").reset_index(drop=True)
    log_rows(logger, claims, "after excluding quarantined claims")
    storms = pd.read_parquet(get_path(cfg, "raw") / "storms_fl.parquet",
                             columns=["EPISODE_ID", "EVENT_ID", "EPISODE_NARRATIVE", "EVENT_NARRATIVE"])

    t_pieces = time.perf_counter()
    raw_pieces = claim_pieces(claims, storms).reindex(claims["id"])
    masked = {p: mask_damage_numbers(p) for p in {p for ps in raw_pieces for p in ps}}  # unique only
    masked_pieces = raw_pieces.map(lambda ps: tuple(dict.fromkeys(masked[p] for p in ps)))
    n_affected = int((raw_pieces != masked_pieces).sum())
    logger.info("Built pieces in %.1f s; masking changes the text of %s claims",
                time.perf_counter() - t_pieces, f"{n_affected:,}")
    has_text = (raw_pieces.map(len) > 0).to_numpy()
    if not np.array_equal(has_text, claims["has_narrative"].to_numpy()):
        raise SystemExit("STOP: claims with narratives in 05 do not match claims with pieces here")
    unique_raw = sorted({p for ps in raw_pieces for p in ps})
    unique_masked = sorted({p for ps in masked_pieces for p in ps})
    long_pieces = sum(len(p.split()) > int(require(cfg, "embedding.chunk_words")) for p in unique_raw)
    logger.info("Unique pieces: %d raw, %d masked (%d split into several chunks)",
                len(unique_raw), len(unique_masked), long_pieces)

    embedder = Embedder(model_name, processed / "embedding_cache.parquet",
                        int(require(cfg, "embedding.batch_size")), int(require(cfg, "embedding.chunk_words")))
    vectors = embedder.piece_vectors(unique_raw + unique_masked)
    embedder.save()
    dim = len(next(iter(vectors.values())))
    t_embed = time.perf_counter() - t0
    logger.info("Embedding done in %.1f s (dimension %d; cache %d -> %d chunks)",
                t_embed, dim, embedder.n_cached_at_start, len(embedder.cache))

    fit_mask = (claims["split"] == "train").to_numpy() & has_text
    logger.info("PCA fit rows (train claims with a narrative): %s of %s", f"{fit_mask.sum():,}", f"{len(claims):,}")
    out = [claims[["id", "split", "has_narrative"]].reset_index(drop=True)]
    variance = []
    for name, pieces, prefix in (("text", raw_pieces, "txt"), ("text_masked", masked_pieces, "txtm")):
        X = claim_matrix(pieces, vectors, dim)
        pca = fit_pca_train_only(X, fit_mask, k, seed)
        joblib.dump({"pca": pca, "fit_ids": claims.loc[fit_mask, "id"].to_numpy(),
                     "model": model_name, "chunk_words": int(require(cfg, "embedding.chunk_words"))},
                    models / f"pca_{name}.joblib")
        ev = pca.explained_variance_ratio_
        logger.info("%-11s PCA %d components explain %.1f%% of variance (PC1 %.1f%%)",
                    name, k, 100 * ev.sum(), 100 * ev[0])
        variance.append(pd.DataFrame({"variant": name, "component": range(1, k + 1),
                                      "explained_variance": ev.round(5),
                                      "cumulative": ev.cumsum().round(5)}))
        out.append(pca_features(pca, X, has_text, prefix))

    features = pd.concat(out, axis=1)
    write_parquet_atomic(features, processed / "text_features.parquet")
    pd.concat(variance).to_csv(get_path(cfg, "tables") / "text_pca_variance.csv", index=False)
    log_rows(logger, features, "text_features.parquet")
    logger.info("Total runtime %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
