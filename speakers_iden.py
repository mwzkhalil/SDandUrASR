# ============================================================
# PRODUCTION-READY HYBRID SPEAKER DIARIZATION + URDU ASR
# Diarization via DiariZen (BUT-FIT); embeddings/ASR via pyannote/transformers
# ============================================================
import os
import sys
import pickle
import subprocess
import signal
import atexit
import gc
import logging

import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from pyannote.audio import Model, Inference
from pyannote.core import Segment, Annotation
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from diarizen.pipelines.inference import DiariZenPipeline

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# GPU MEMORY SAFETY
# ============================================================
def emergency_gpu_cleanup():
    """Aggressive GPU memory cleanup"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        logger.info("GPU Memory: %.2fGB allocated, %.2fGB reserved, %.2fGB free of %.2fGB total", allocated, reserved, free, total)
        return free

atexit.register(emergency_gpu_cleanup)
signal.signal(signal.SIGINT, lambda s,f: (emergency_gpu_cleanup(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s,f: (emergency_gpu_cleanup(), sys.exit(0)))

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_FILE = "/mnt/4ED699D7D699C01F/goodmorningshowrecording/Muskurati Subha With Zeeshan Azhar Good Morning Pakistan Part-1 Metro1 News 2 Fe_20260206_164615.mp4"
OUTPUT_TXT = "transcription_final1.txt"
SPEAKER_DB = "speaker_embeddings1.pkl"
CHUNK_LENGTH_SEC = 10*60
MIN_SEG_DUR = 1.2
MIN_DUR_MATCH = 2.5
MERGE_GAP_TOLERANCE = 0.8
MIN_SPEAKERS_PER_CHUNK = 2
MAX_SPEAKERS_PER_CHUNK = 12
MATCH_THRESHOLD = 0.65
CLUSTER_DISTANCE_THRESHOLD = 0.35
MIN_CLUSTER_SIZE = 3
USE_MULTI_PASS_CLUSTERING = True
ENERGY_WEIGHTED_THRESHOLD = 10.0
DIARIZATION_MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
WINDOW_DUR = 3.0
WINDOW_STEP = 1.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_FULL_WAV = "temp_full.wav"
TEMP_CLEAN_WAV = "temp_clean.wav"
TEMP_SEG_WAV = "temp_seg.wav"
ASR_MODEL_ID = "sajadkawa/ns_finetune_urdu_asr_org"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def normalize(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-9 else x

def seconds_to_hhmmss(sec):
    sec = int(sec)
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"

# ============================================================
# LOAD MODELS
# ============================================================
logger.info("Loading diarization model (DiariZen)...")
diar_pipeline = DiariZenPipeline.from_pretrained(DIARIZATION_MODEL_ID)
check_gpu_memory()

logger.info("Loading embedding model...")
embedding_model = Model.from_pretrained("pyannote/embedding")
embedding_model.eval()
embedding_inference = Inference(embedding_model, window="whole", device=DEVICE)
check_gpu_memory()

logger.info("Loading Urdu ASR: %s", ASR_MODEL_ID)
processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    ASR_MODEL_ID,
    torch_dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True if DEVICE.type=="cuda" else False
)
emergency_gpu_cleanup()
asr_model = asr_model.to(DEVICE)
check_gpu_memory()

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if DEVICE.type=="cuda" else -1,
    chunk_length_s=30,
    generate_kwargs={"language":"ur", "task":"transcribe"}
)

# ============================================================
# SPEAKER DATABASE
# ============================================================
if os.path.exists(SPEAKER_DB):
    with open(SPEAKER_DB, "rb") as f:
        speaker_db = pickle.load(f)
    for k, v in list(speaker_db.items()):
        if not isinstance(v, list):
            speaker_db[k] = [v]
    logger.info("Loaded speaker DB with %d speakers", len(speaker_db))
else:
    speaker_db = {}
    logger.info("Creating new speaker database")

# ============================================================
# EMBEDDING EXTRACTION
# ============================================================
def extract_embedding(audio_path, segment, dur):
    try:
        if dur < ENERGY_WEIGHTED_THRESHOLD:
            emb = embedding_inference.crop(audio_path, segment)
            return normalize(emb)
       
        waveform, sr = sf.read(audio_path)
        embs, energies = [], []
        t = segment.start
        while t + WINDOW_DUR <= segment.end:
            sub = Segment(t, t+WINDOW_DUR)
            emb = embedding_inference.crop(audio_path, sub)
            s = int(t*sr); e=int((t+WINDOW_DUR)*sr)
            window_audio = waveform[s:e]
            energy = np.sqrt(np.mean(window_audio**2))
            if energy > 0.01:
                embs.append(emb)
                energies.append(energy)
            t += WINDOW_STEP
        if len(embs)==0:
            emb = embedding_inference.crop(audio_path, segment)
            return normalize(emb)
        weights = np.array(energies)/(np.sum(energies)+1e-6)
        emb = np.sum(np.array(embs)*weights[:,None],axis=0)
        return normalize(emb)
    except Exception as e:
        logger.warning("Embedding extraction failed: %s", e, exc_info=True)
        return None

def find_best_match(emb, db, min_samples=3):
    best_id, best_sim = None, -1
    for sid,lst in db.items():
        if len(lst)<min_samples:
            centroid = normalize(np.mean(lst,axis=0))
            sim = 1 - cosine(emb, centroid)
        else:
            sims = [1 - cosine(emb,e) for e in lst[-10:]]
            sim = np.median(sims)
        if sim>best_sim: best_sim=sim; best_id=sid
    return (best_id,best_sim) if best_sim>=MATCH_THRESHOLD else (None,best_sim)

# ============================================================
# MULTI-PASS CLUSTERING
# ============================================================
def adaptive_clustering(embs, min_clusters=2, max_clusters=15):
    best_labels = None; best_score = -1; best_n = 0
    for thresh in [0.30,0.35,0.40,0.45]:
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh, metric="cosine", linkage="average")
        labels = clusterer.fit_predict(embs)
        n_clusters = len(set(labels))
        if n_clusters<min_clusters or n_clusters>max_clusters: continue
        try:
            score = silhouette_score(embs, labels, metric='cosine') if n_clusters > 1 else 0
        except Exception:
            score = 0
        if score>best_score: best_score=score; best_labels=labels; best_n=n_clusters
    if best_labels is None:
        best_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTER_DISTANCE_THRESHOLD, metric="cosine", linkage="average").fit_predict(embs)
    logger.info("Selected %d clusters (silhouette=%.3f)", best_n, best_score)
    return best_labels

# ============================================================
# AUDIO EXTRACTION & CLEANUP
# ============================================================
logger.info("Preparing audio...")
subprocess.run(["ffmpeg", "-y", "-i", VIDEO_FILE, "-vn", "-ac", "1", "-ar", "16000", TEMP_FULL_WAV], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["ffmpeg", "-y", "-i", TEMP_FULL_WAV, "-af", "highpass=f=100,lowpass=f=6000,afftdn=nf=-20,loudnorm,speechnorm", TEMP_CLEAN_WAV], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
audio = AudioSegment.from_wav(TEMP_CLEAN_WAV)
total_dur = len(audio) / 1000
logger.info("Audio duration: %s", seconds_to_hhmmss(total_dur))

# ============================================================
# CHUNKED SPEAKER DIARIZATION (DiariZen – itertracks API)
# ============================================================
full_ann = Annotation()
offset = 0.0
num_chunks = (len(audio) + CHUNK_LENGTH_SEC * 1000 - 1) // (CHUNK_LENGTH_SEC * 1000)

for i, ms in enumerate(range(0, len(audio), CHUNK_LENGTH_SEC * 1000)):
    chunk = audio[ms : ms + CHUNK_LENGTH_SEC * 1000]
    dur = len(chunk) / 1000.0
    path = f"chunk_{i}.wav"
    chunk.export(path, "wav")
    emergency_gpu_cleanup()

    try:
        diar_results = diar_pipeline(path)
        speakers_in_chunk = set()
        for turn, _, speaker in diar_results.itertracks(yield_label=True):
            shifted = Segment(turn.start + offset, turn.end + offset)
            full_ann[shifted] = f"chunk{i}_{speaker}"
            speakers_in_chunk.add(speaker)
        logger.info("Chunk %d/%d: %d speakers", i + 1, num_chunks, len(speakers_in_chunk))
    except Exception as e:
        logger.exception("Chunk %d diarization error: %s", i, e)

    offset += dur
    if os.path.exists(path):
        os.remove(path)
    emergency_gpu_cleanup()

# ============================================================
# MERGE SPEAKER TURNS
# ============================================================
turns = [(t.start, t.end, lbl) for t, _, lbl in full_ann.itertracks(yield_label=True)]
turns.sort(key=lambda x: x[0])

if not turns:
    logger.error("No speaker turns found after diarization. Check audio / model output.")
    sys.exit(1)

merged = []
cur_s, cur_e, cur_lbl = turns[0]
for ns, ne, nlbl in turns[1:]:
    if nlbl == cur_lbl and ns <= cur_e + MERGE_GAP_TOLERANCE:
        cur_e = max(cur_e, ne)
    else:
        if cur_e - cur_s >= MIN_SEG_DUR:
            merged.append((cur_s, cur_e, cur_lbl))
        cur_s, cur_e, cur_lbl = ns, ne, nlbl

if cur_e - cur_s >= MIN_SEG_DUR:
    merged.append((cur_s, cur_e, cur_lbl))

logger.info("Merged segments: %d", len(merged))

# The rest of your code remains **exactly** the same from here
# ============================================================
# EXTRACT EMBEDDINGS + ADAPTIVE CLUSTERING
# ============================================================
segment_data=[]
for i,(s,e,lbl) in enumerate(merged):
    dur=e-s
    emb=extract_embedding(TEMP_CLEAN_WAV, Segment(s,e), dur) if dur>=MIN_DUR_MATCH else None
    segment_data.append((i,s,e,dur,emb,lbl))

idxs=[i for i,x in enumerate(segment_data) if x[4] is not None]
embs=np.array([segment_data[i][4] for i in idxs])
logger.info("Valid embeddings: %d/%d", len(embs), len(segment_data))

labels=adaptive_clustering(embs) if USE_MULTI_PASS_CLUSTERING and len(embs)>10 else AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTER_DISTANCE_THRESHOLD, metric="cosine", linkage="average").fit_predict(embs)

idx_to_cluster={idxs[i]:labels[i] for i in range(len(labels))}
cluster_to_speaker={}
cluster_counts={l:sum(labels==l) for l in set(labels)}

for cid in set(labels):
    mask = labels==cid
    cluster_embs = embs[mask]
    centroid=normalize(np.mean(cluster_embs,axis=0))
    sid,sim=find_best_match(centroid,speaker_db)
    if sid and cluster_counts[cid]>=MIN_CLUSTER_SIZE:
        speaker_db[sid].append(centroid)
        for emb in cluster_embs[:5]: speaker_db[sid].append(emb)
        cluster_to_speaker[cid]=sid
    elif cluster_counts[cid]>=MIN_CLUSTER_SIZE:
        sid=f"Speaker_{len(speaker_db)+1}"; speaker_db[sid]=[centroid]+list(cluster_embs[:5])
        cluster_to_speaker[cid]=sid
    else:
        if speaker_db:
            sid,sim=find_best_match(centroid,speaker_db,min_samples=1)
            cluster_to_speaker[cid]=sid if sid and sim>MATCH_THRESHOLD-0.1 else f"Speaker_{len(speaker_db)+1}"
        else:
            sid=f"Speaker_{len(speaker_db)+1}"; speaker_db[sid]=[centroid]; cluster_to_speaker[cid]=sid

logger.info("Identified %d unique speakers", len(set(cluster_to_speaker.values())))

# ============================================================
# TRANSCRIPTION
# ============================================================
lines=[]
for idx,(i,s,e,dur,emb,lbl) in enumerate(segment_data):
    if idx % 10 == 0: emergency_gpu_cleanup()
    subprocess.run(["ffmpeg","-y","-i",TEMP_CLEAN_WAV,"-ss",str(s),"-to",str(e),TEMP_SEG_WAV],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        text = asr_pipe(TEMP_SEG_WAV)["text"].strip() or "[no speech detected]"
    except Exception as e:
        logger.warning("Transcription failed for segment %d: %s", idx, e, exc_info=True)
        text = "[transcription failed]"
    spk=cluster_to_speaker.get(idx_to_cluster.get(i),"UNK")
    lines.append((s,e,spk,text))

# ============================================================
# SAVE RESULTS
# ============================================================
with open(SPEAKER_DB, "wb") as f:
    pickle.dump(speaker_db, f)
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"# Urdu transcription using {ASR_MODEL_ID}\n")
    f.write(f"# Duration: {seconds_to_hhmmss(total_dur)}\n")
    f.write(f"# Speakers: {len(set(x[2] for x in lines))}\n")
    f.write(f"# Segments: {len(lines)}\n")
    f.write("="*80+"\n\n")
    for s,e,spk,text in lines:
        f.write(f"[{seconds_to_hhmmss(s)} → {seconds_to_hhmmss(e)}] {spk}: {text}\n")

# ============================================================
# STATISTICS
# ============================================================
unique_speakers=set(x[2] for x in lines)
logger.info("Speakers detected: %s", ", ".join(sorted(unique_speakers)))

speaker_time={}; speaker_segs={}
for s,e,spk,_ in lines:
    speaker_time[spk]=speaker_time.get(spk,0)+(e-s)
    speaker_segs[spk]=speaker_segs.get(spk,0)+1

logger.info("Speaker talk time:")
for spk in sorted(speaker_time.keys()):
    dur = speaker_time[spk]
    segs = speaker_segs[spk]
    pct = (dur / total_dur) * 100
    avg_seg = dur / segs if segs > 0 else 0
    logger.info("  %s: %s (%.1f%%) - %d segments (avg %.1fs)", spk, seconds_to_hhmmss(dur), pct, segs, avg_seg)

# ============================================================
# CLEANUP
# ============================================================
for f in [TEMP_FULL_WAV,TEMP_CLEAN_WAV,TEMP_SEG_WAV]:
    if os.path.exists(f): os.remove(f)

emergency_gpu_cleanup()
logger.info("Process complete")