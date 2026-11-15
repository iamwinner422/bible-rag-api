import gzip
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class E5Embedder:
    def __init__(self, model_dir=None):
        project_root = Path(__file__).resolve().parents[2]
        model_path = (
            Path(model_dir) if model_dir is not None else project_root / "model"
        )
        if not model_path.is_absolute():
            model_path = project_root / model_path

        with gzip.open(
            model_path / "tokenizer.json.gz", "rt", encoding="utf-8"
        ) as file:
            self.tokenizer = Tokenizer.from_str(file.read())
        self.tokenizer.enable_truncation(512)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        self.session = ort.InferenceSession(
            str(model_path / "model.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )

        # Noms d'entrées du modèle ONNX
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def _mean_pooling(self, token_embeddings, attention_mask):
        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.maximum(mask.sum(axis=1), 1e-9)
        return summed / counts

    def embed(self, texts, batch_size=32, is_query=False):
        """
        is_query=True  → préfixe "query: "
        is_query=False → préfixe "passage: "
        Retourne des vecteurs normalisés (float32).
        """
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + (t or "") for t in texts]

        parts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = self.tokenizer.encode_batch(batch)

            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            feeds = {}
            if "input_ids" in self.input_names:
                feeds["input_ids"] = ids
            if "attention_mask" in self.input_names:
                feeds["attention_mask"] = mask
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)

            outputs = self.session.run(None, feeds)
            token_embeddings = np.asarray(outputs[0], dtype=np.float32)

            vecs = self._mean_pooling(token_embeddings, mask)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
            parts.append(vecs)

        return np.vstack(parts)
