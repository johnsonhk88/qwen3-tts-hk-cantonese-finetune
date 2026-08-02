# coding=utf-8
# cluster_speakers.py
# Automatic Speaker Identification and Diarization for Qwen3-TTS Fine-Tuning Dataset.

import os
import sys
import json
import torch
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure scikit-learn is installed
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize
except ImportError:
    print("scikit-learn is not installed. Installing it now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

def main():
    audio_dir = "./audio"
    parquet_path = "train-00000-of-00001.parquet"
    output_jsonl = "train_raw.jsonl"
    
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found. Please ensure you are in the correct directory.")
        sys.exit(1)
        
    print(f"Loading raw dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Check what audio files exist in the audio folder
    print("Scanning audio folder for existing resampled files...")
    existing_wavs = []
    for file in os.listdir(audio_dir):
        if file.endswith(".wav") and file[:-4].isdigit():
            existing_wavs.append(int(file[:-4]))
            
    existing_wavs = sorted(existing_wavs)
    print(f"Found {len(existing_wavs)} valid wav files in {audio_dir}.")
    
    if len(existing_wavs) == 0:
        print("Error: No resampled wav files found in ./audio directory. Please run the resample cells first.")
        sys.exit(1)
        
    # Load SpeechBrain Speaker Encoder
    print("🔄 Loading ECAPA-TDNN speaker encoder...")
    # Monkey-patch torchaudio if needed (for newer torchaudio versions)
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
        
    from speechbrain.pretrained import EncoderClassifier
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
    )
    
    # Extract speaker embeddings
    embeddings = []
    valid_indices = []
    
    print("🎙️ Extracting speaker embeddings...")
    for idx in tqdm(existing_wavs):
        audio_path = os.path.join(audio_dir, f"{idx}.wav")
        try:
            # Use soundfile instead of torchaudio.load to avoid TorchCodec errors
            data, fs = sf.read(audio_path)
            if len(data.shape) > 1:
                data = data.mean(axis=-1)
            signal = torch.from_numpy(data).float().unsqueeze(0)
            
            if fs != 16000:
                signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)
                
            with torch.no_grad():
                emb = speaker_encoder.encode_batch(signal.to(speaker_encoder.device))
                emb = emb.squeeze().cpu().numpy()
                
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Failed to extract embedding for {audio_path}: {e}")
            
    if len(embeddings) == 0:
        print("Error: Failed to extract any speaker embeddings.")
        sys.exit(1)
        
    print(f"Extracted {len(embeddings)} speaker embeddings successfully.")
    
    # L2-normalize embeddings for cosine distance clustering
    embeddings_norm = normalize(np.array(embeddings), norm='l2')
    
    # Run Agglomerative Clustering
    # distance_threshold is cosine distance (1 - cosine_similarity).
    # 0.25 threshold corresponds to 0.75 cosine similarity. Group similarity above 75%.
    print("🧩 Grouping similar voices together using Cosine Distance Clustering...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=0.30
    )
    labels = clustering.fit_predict(embeddings_norm)
    
    num_speakers = len(set(labels))
    print(f"🎉 Discovered {num_speakers} unique speakers in your dataset!")
    
    # Build mapping dictionary
    speaker_mapping = {idx: f"spk_{label:03d}" for idx, label in zip(valid_indices, labels)}
    
    # Slice the original dataframe for the valid processed indexes
    selected_df = df.iloc[valid_indices].copy()
    selected_df['speaker_id'] = selected_df.index.map(speaker_mapping)
    selected_df['audio_path'] = [f"./audio/{idx}.wav" for idx in valid_indices]
    
    # Create speaker-matched ref_audio path
    # We will pick the first wav of each speaker as their designated ref_audio
    print("🔗 Creating speaker-matched reference mappings...")
    ref_audio_map = {}
    for spk_id, group in selected_df.groupby('speaker_id'):
        ref_audio_map[spk_id] = group.iloc[0]['audio_path']
        
    selected_df['ref_audio'] = selected_df['speaker_id'].map(ref_audio_map)
    
    # Write updated train_raw.jsonl
    print(f"💾 Saving speaker-matched training data to {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for i, row in selected_df.iterrows():
            obj = {
                "audio": row["audio_path"],
                "text": row["sentence"],
                "ref_audio": row["ref_audio"],
                "speaker_id": row["speaker_id"]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
    print(f"✅ Saved! {output_jsonl} now has {len(selected_df)} speaker-matched entries.")
    
    # Display speaker distribution
    print("\nTop 10 speakers by audio count:")
    counts = selected_df['speaker_id'].value_counts()
    print(counts.head(10))

if __name__ == "__main__":
    main()