import os, time, logging, tempfile
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import soundfile as sf

logger = logging.getLogger("blitz.services.diarization")

@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float
    text: Optional[str] = None

@dataclass
class DiarizationResult:
    segments: List[SpeakerSegment]
    num_speakers: int
    processing_time_sec: float
    method: str = "nemo-clustering"

class DiarizationService:
    def __init__(self, device="cuda", max_speakers=10):
        self.device = device
        self.max_speakers = max_speakers
        self._titanet_model = None
        self._available = False
        self._load_models()

    def _load_models(self):
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
            import torch
            logger.info("Loading TitaNet-Large for speaker embeddings...")
            start = time.time()
            self._titanet_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
            self._titanet_model.eval()
            if self.device == "cuda" and torch.cuda.is_available():
                self._titanet_model = self._titanet_model.to("cuda")
            logger.info(f"TitaNet-Large loaded ({time.time()-start:.1f}s)")
            self._available = True
        except Exception as e:
            logger.error(f"Failed to load diarization models: {e}")

    @property
    def is_available(self):
        return self._available

    def diarize(self, audio, sample_rate=16000, num_speakers=None, min_speakers=None, max_speakers=None):
        if not self.is_available:
            raise RuntimeError("Diarization model not available.")
        import torch
        from sklearn.cluster import AgglomerativeClustering
        start_time = time.time()
        max_spk = num_speakers or max_speakers or self.max_speakers
        window_sec, shift_sec = 1.5, 0.75
        window_samples = int(window_sec * sample_rate)
        shift_samples = int(shift_sec * sample_rate)
        windows, window_starts = [], []
        for start_idx in range(0, len(audio) - window_samples + 1, shift_samples):
            window = audio[start_idx:start_idx + window_samples]
            if np.abs(window).mean() > 0.001:
                windows.append(window)
                window_starts.append(start_idx / sample_rate)
        if len(windows) < 2:
            return DiarizationResult(segments=[SpeakerSegment(speaker="speaker_0", start=0, end=len(audio)/sample_rate)], num_speakers=1, processing_time_sec=time.time()-start_time)
        embeddings = []
        for i in range(0, len(windows), 16):
            batch = windows[i:i+16]
            temp_files = []
            for w in batch:
                f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(f.name, w, sample_rate)
                temp_files.append(f.name)
                f.close()
            try:
                with torch.no_grad():
                    batch_embs = []
                    for tf in temp_files:
                        emb = self._titanet_model.get_embedding(tf)
                        if isinstance(emb, torch.Tensor):
                            batch_embs.append(emb.cpu().numpy().reshape(1,-1))
                        else:
                            batch_embs.append(np.array(emb).reshape(1,-1))
                    embeddings.append(np.vstack(batch_embs))
            finally:
                for tf in temp_files:
                    os.unlink(tf)
        all_embeddings = np.vstack(embeddings)
        n_clusters = num_speakers or min(max_spk, max(2, min(len(windows)//10, 8)))
        if min_speakers and n_clusters < min_speakers:
            n_clusters = min_speakers
        try:
            labels = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(all_embeddings) if n_clusters >= 2 else np.zeros(len(all_embeddings), dtype=int)
        except:
            labels = np.zeros(len(all_embeddings), dtype=int)
        raw_segs = [SpeakerSegment(speaker=f"speaker_{l}", start=round(s,3), end=round(s+window_sec,3)) for s, l in zip(window_starts, labels)]
        merged = self._merge_segments(raw_segs)
        return DiarizationResult(segments=merged, num_speakers=len(set(labels)), processing_time_sec=round(time.time()-start_time, 3))

    def _merge_segments(self, segments):
        if not segments: return []
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.speaker == merged[-1].speaker and (seg.start - merged[-1].end) < 0.5:
                merged[-1] = SpeakerSegment(speaker=merged[-1].speaker, start=merged[-1].start, end=seg.end)
            else:
                merged.append(seg)
        return merged

    def merge_with_transcription(self, diar_result, word_timestamps, full_text=None):
        if not word_timestamps:
            # Fallback: split full text proportionally across segments by duration
            if full_text and diar_result.segments:
                words = full_text.split()
                total_duration = sum(s.end - s.start for s in diar_result.segments)
                if total_duration > 0:
                    word_idx = 0
                    for seg in diar_result.segments:
                        seg_duration = seg.end - seg.start
                        seg_word_count = max(1, int(len(words) * (seg_duration / total_duration)))
                        seg.text = " ".join(words[word_idx:word_idx + seg_word_count])
                        word_idx += seg_word_count
                    # Assign remaining words to last segment
                    if word_idx < len(words) and diar_result.segments:
                        last = diar_result.segments[-1]
                        last.text = (last.text + " " + " ".join(words[word_idx:])).strip()
                return diar_result.segments
            return diar_result.segments
        speaker_words = []
        for w in word_timestamps:
            mid = (w.get("start",0) + w.get("end",0)) / 2
            best = "speaker_0"
            for seg in diar_result.segments:
                if seg.start <= mid <= seg.end:
                    best = seg.speaker; break
            speaker_words.append((best, w.get("word",""), w.get("start",0), w.get("end",0)))
        merged = []
        cur_spk, cur_words, cur_s, cur_e = None, [], 0, 0
        for spk, word, s, e in speaker_words:
            if spk != cur_spk:
                if cur_words: merged.append(SpeakerSegment(speaker=cur_spk, start=round(cur_s,3), end=round(cur_e,3), text=" ".join(cur_words)))
                cur_spk, cur_words, cur_s, cur_e = spk, [word], s, e
            else:
                cur_words.append(word); cur_e = e
        if cur_words: merged.append(SpeakerSegment(speaker=cur_spk, start=round(cur_s,3), end=round(cur_e,3), text=" ".join(cur_words)))
        return merged

def format_diarized_transcript(segments):
    return "\n".join(f"[{int(s.start//60):02d}:{s.start%60:04.1f} - {int(s.end//60):02d}:{s.end%60:04.1f}] {s.speaker}: {s.text or ''}" for s in segments)

def format_diarized_json(segments):
    return [{"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text} for s in segments]
