from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_ID = "intfloat/multilingual-e5-small"
OUTPUT_DIR = Path("model_e5")

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

    print(f"✅ Modèle exporté dans {OUTPUT_DIR}/")
    print("Fichiers attendus : model.onnx + tokenizer.json (+ config)")

if __name__ == "__main__":
    main()