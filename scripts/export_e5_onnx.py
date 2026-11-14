import gzip
from pathlib import Path
import shutil
from optimum.onnxruntime import ORTModelForFeatureExtraction
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

MODEL_ID = "intfloat/multilingual-e5-small"
OUTPUT_DIR = Path("model")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Téléchargement et export de {MODEL_ID}...")
    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_ID,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    compressed_tokenizer_path = OUTPUT_DIR / "tokenizer.json.gz"
    with tokenizer_path.open("rb") as source, gzip.open(compressed_tokenizer_path, "wb", compresslevel=9) as target:
        shutil.copyfileobj(source, target)
    tokenizer_path.unlink()

    model_path = OUTPUT_DIR / "model.onnx"
    quantized_path = OUTPUT_DIR / "model.quantized.onnx"
    quantize_dynamic(
        str(model_path),
        str(quantized_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    quantized_path.replace(model_path)

    print(f"Modèle E5 exporté et quantifié en INT8 dans {OUTPUT_DIR}/")
    print(f"Taille du modèle ONNX : {model_path.stat().st_size / 1_000_000:.1f} MB")

if __name__ == "__main__":
    main()