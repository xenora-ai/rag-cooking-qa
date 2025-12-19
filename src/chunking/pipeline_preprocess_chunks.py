# src/chunking/pipeline_preprocess_chunks.py
import os
from preprocess import clean_text
from splitter import split_into_chunks


RAW_DIR = "../../data/raw"
PROCESSED_DIR = "../../data/processed"
CHUNKS_DIR = "../../data/chunks"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if filename.endswith(".txt"):
        raw_path = os.path.join(RAW_DIR, filename)
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 1. Очистка
        clean = clean_text(raw_text)

        # Збереження очищеного тексту
        processed_path = os.path.join(PROCESSED_DIR, filename)
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(clean)

        # 2. Чанкінг
        chunks = split_into_chunks(clean, max_len=200, overlap=50)

        # Збереження чанків
        for i, chunk in enumerate(chunks, start=1):
            chunk_filename = filename.replace(".txt", f"_chunk{i:02d}.txt")
            chunk_path = os.path.join(CHUNKS_DIR, chunk_filename)
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk)

        print(f"Processed {filename}: {len(chunks)} chunks")
