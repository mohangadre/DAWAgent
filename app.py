import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
from pathlib import Path
import pretty_midi
from datetime import datetime
from pydub import AudioSegment
import logging
import traceback
from collections import Counter
from scipy.signal import find_peaks
import soundfile as sf

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configure ffmpeg path for pydub
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffmpeg = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"

# Instead of hardcoding the ffmpeg path, we could also implement a check to see if ffmpeg is available in the system PATH and use it directly. Right now, running this locally causes no issues, but would break on some machines upon deployment, specifically for MP3 and M4A files. WAV files remain unaffected as they skip the pydub conversion.
# Using a dynamic configuration approach would make the code more portable across different environments without requiring users to modify the code for their specific ffmpeg installation.

'''
import shutil

def _configure_ffmpeg():
    candidate = shutil.which("ffmpeg")
    
    if candidate is None:
        fallback = [
            "/opt/homebrew/bin/ffmpeg", # macOS for older Apple machines
            "/usr/local/bin/ffmpeg",    # Intel or manual installations
            "/usr/bin/ffmpeg",          # Linux
        ]
        
        for path in fallback:
            if os.path.isfile(path):
                candidate = path
                break
    
    if candidate is None:
        logger.warning(
            "ffmpeg not found. MP3 and M4A files will not be supported."
            "Install ffmpeg and ensure it's in your system PATH."
        )
        return
    
    AudioSegment.converter = candidate
    AudioSegment.ffmpeg = candidate
    
    ffprobe = shutil.which("ffprobe") or candidate.replace("ffmpeg", "ffprobe")
    AudioSegment.ffprobe = ffprobe
    
    logger.info(f"ffmpeg configured: {candidate}")

_configure_ffmpeg()
'''


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def midi_to_note_name(midi_number):
   """Convert MIDI note number to note name (e.g. 60 -> 'C3').
   Uses the DAW-standard convention where MIDI 60 = C3 (matching
   most piano-roll displays like Ableton, FL Studio, etc.)."""
   if midi_number < 0 or midi_number > 127:
       return "N/A"
   notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
   octave = (midi_number // 12) - 2
   note = notes[midi_number % 12]
   return f"{note}{octave}"




def identify_chord(midi_notes):
   """
   Identify chord type from a list of MIDI note numbers.
   Returns (root_note_name, chord_type) or (None, None) if not recognised.
   """
   if len(midi_notes) < 2:
       return None, None


   notes = sorted(set(int(n) for n in midi_notes))
   root = notes[0]
   intervals_normalized = sorted(set((n - root) % 12 for n in notes))


   chord_patterns = {
       # Triads
       (0, 4, 7): "major",
       (0, 3, 7): "minor",
       (0, 3, 6): "diminished",
       (0, 4, 8): "augmented",
       (0, 2, 7): "sus2",
       (0, 5, 7): "sus4",
       # 7th chords
       (0, 4, 7, 11): "maj7",
       (0, 4, 7, 10): "7",
       (0, 3, 7, 10): "min7",
       (0, 3, 6, 9): "dim7",
       (0, 3, 6, 10): "m7b5",
       # Extended / 9th
       (0, 4, 7, 10, 14): "9",
       (0, 4, 7, 11, 14): "maj9",
       (0, 3, 7, 10, 14): "min9",
       # 6th chords
       (0, 4, 7, 9): "6",
       (0, 3, 7, 9): "min6",
       # Add chords
       (0, 4, 7, 14): "add9",
       (0, 3, 7, 14): "minadd9",
   }


   key = tuple(intervals_normalized)
   if key in chord_patterns:
       return midi_to_note_name(root), chord_patterns[key]
   if len(intervals_normalized) == 2 and 7 in intervals_normalized:
       return midi_to_note_name(root), "5"
   if len(notes) >= 2:
       return midi_to_note_name(root), f"chord({len(notes)} notes)"
   return None, None




# ---------------------------------------------------------------------------
# Frame-by-frame polyphonic note tracking
# ---------------------------------------------------------------------------


# Harmonic intervals in semitones to filter (2nd–8th harmonics)
HARMONIC_INTERVALS = {12, 19, 24, 28, 31, 34, 36}




def detect_notes_framewise(y, sr):
   """
   Detect all note events by tracking pitches frame-by-frame across the
   full CQT spectrogram.  Each pitch is independently tracked for when it
   starts and when it stops, so overlapping notes (e.g. a sustained chord
   with melody notes on top) are handled naturally.


   Key improvements:
   - Octave-aware harmonic filter: when two peaks are an octave apart,
     always keep the lower (fundamental) and remove the higher (harmonic).
   - Onset-based re-articulation: notes that span across a strong onset
     (chord change) are split so common tones get re-struck.
   - pyin merge for single-note frames.


   Returns a list of note dicts sorted by (start, pitch):
       [{'pitch': int, 'start': float, 'end': float, 'velocity_raw': int}, ...]
   """
   fmin_hz = librosa.note_to_hz('C2')   # 65.41 Hz
   fmin_midi = int(round(librosa.hz_to_midi(fmin_hz)))  # MIDI 36
   n_bins = 84                          # 7 octaves (C2–B8)
   hop = 512


   # --- Full-track CQT ----------------------------------------------------
   logger.info("Computing full-track CQT...")
   C = np.abs(librosa.cqt(
       y, sr=sr, fmin=fmin_hz,
       n_bins=n_bins, bins_per_octave=12, hop_length=hop
   ))
   num_frames = C.shape[1]
   cqt_times = librosa.frames_to_time(np.arange(num_frames), sr=sr, hop_length=hop)
   logger.info(f"CQT: {C.shape[0]} bins x {num_frames} frames, "
               f"time range 0–{cqt_times[-1]:.3f}s")


   # --- Per-frame RMS energy for noise gating ------------------------------
   frame_rms = np.zeros(num_frames)
   for t in range(num_frames):
       start_samp = t * hop
       end_samp = min(start_samp + hop, len(y))
       seg = y[start_samp:end_samp]
       if len(seg) > 0:
           frame_rms[t] = float(np.sqrt(np.mean(seg ** 2)))
   global_rms = float(np.sqrt(np.mean(y ** 2)))
   noise_floor = global_rms * 0.08  # 8% of global RMS as noise floor (tunable)
   logger.info(f"RMS noise floor: {noise_floor:.6f}  (global RMS={global_rms:.6f})")


   # --- Full-track pyin (monophonic aid) -----------------------------------
   logger.info("Running pyin for monophonic aid...")
   f0, voiced_flag, voiced_probs = librosa.pyin(
       y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'),
       sr=sr, frame_length=4096, hop_length=hop
   )
   logger.info(f"pyin: {np.sum(voiced_flag)} voiced / {len(f0)} frames")


   # --- Detect active pitches per CQT frame --------------------------------
   logger.info("Analysing CQT frames...")
   frame_notes = []  # list[set[int]]  – active MIDI notes per frame
   frame_octave_rejections = []  # list[list[(rejected, keeper)]] – HF octave rejections
   global_cqt_max = float(C.max()) if C.max() > 0 else 1.0


   for t in range(num_frames):
       # Gate: skip silent / very weak frames
       if frame_rms[t] < noise_floor:
           frame_notes.append(set())
           frame_octave_rejections.append([])
           continue
       col = C[:, t]
       col_max = col.max()
       if col_max == 0 or col_max < global_cqt_max * 0.03: # 3% of global max as another gate
           frame_notes.append(set())
           frame_octave_rejections.append([])
           continue


       col_norm = col / col_max


       peaks, _ = find_peaks(
           col_norm,
           height=0.12,       # 12 % of frame max
           distance=2,        # ≥ 2 semitones apart
           prominence=0.06
       )
       if len(peaks) == 0:
           frame_notes.append(set())
           frame_octave_rejections.append([])
           continue


       # Candidates sorted strongest-first for initial processing
       candidates = sorted(
           [(fmin_midi + p, float(col_norm[p])) for p in peaks],
           key=lambda x: x[1], reverse=True
       )


       # --- Harmonic filter (stronger wins) --------------------------------
       # Candidates are sorted strongest-first.  When two candidates are
       # related by a harmonic interval (octave, octave+5th, …), the
       # weaker one is rejected.  Exception: bass octave doubles (lower
       # note below bass cutoff): allow both at octave intervals so that
       # sustained bass notes survive alongside their upper harmonics.
       BASS_OCTAVE_CUTOFF = 60   # MIDI 60 = C3 (DAW) – below = bass register
       BASS_OCTAVE_INTERVALS = {12, 24, 36}  # 1/2/3 octave intervals
       fundamentals = []
       hf_kept = []
       hf_rejected = []
       for midi_note, energy in candidates:
           is_harm = False
           rejected_by = None
           for fm, fe in fundamentals:
               interval = abs(midi_note - fm)
               if interval not in HARMONIC_INTERVALS:
                   continue
               if (interval in BASS_OCTAVE_INTERVALS
                       and min(midi_note, fm) < BASS_OCTAVE_CUTOFF):
                   continue
               is_harm = True
               rejected_by = fm
               break
           if not is_harm and 21 <= midi_note <= 108:
               fundamentals.append((midi_note, energy))
               hf_kept.append((midi_note, energy))
           elif is_harm:
               hf_rejected.append((midi_note, energy, rejected_by))
           if len(fundamentals) >= 6:
               break

       # Compact HF log: one summary line per frame (every 10th + first 5)
       if t % 10 == 0 or t <= 5:
           kept_str = ", ".join(
               f"{midi_to_note_name(m)}({energy:.3f})" for m, energy in hf_kept)
           rej_str = ", ".join(
               f"{midi_to_note_name(m)}({e:.3f})<-{midi_to_note_name(rb)}"
               for m, e, rb in hf_rejected)
           logger.info(
               f"  [HF] f={t} ({cqt_times[t]:.2f}s) "
               f"KEPT=[{kept_str}] REJ=[{rej_str}]")

       # --- Bass octave fallback: A3 may not form a peak when A2 is strong ---
       bass_fallback_added = []
       for midi_note, _ in list(fundamentals):
           if midi_note < BASS_OCTAVE_CUTOFF and midi_note + 12 <= 108:
               octave_up = midi_note + 12
               if octave_up not in (m for m, _ in fundamentals):
                   bin_idx = octave_up - fmin_midi
                   if 0 <= bin_idx < len(col_norm):
                       val = float(col_norm[bin_idx])
                       if val > 0.005:
                           fundamentals.append((octave_up, val))
                           bass_fallback_added.append(
                               f"{midi_to_note_name(octave_up)}({val:.3f})"
                               f" from {midi_to_note_name(midi_note)}")

       final_set = set(m for m, _ in fundamentals)
       frame_notes.append(final_set)
       frame_octave_rejections.append(
           [(m, rb) for m, e, rb in hf_rejected if abs(m - rb) == 12])

       if (t % 10 == 0 or t <= 5) and bass_fallback_added:
           logger.info(
               f"  [BF] f={t} bass-fallback added: "
               f"{', '.join(bass_fallback_added)}")
       if t % 10 == 0 or t <= 5:
           final_str = ", ".join(sorted(
               midi_to_note_name(m) for m in final_set))
           logger.info(f"  [FRAME] f={t} ({cqt_times[t]:.2f}s) notes: [{final_str}]")


   # --- Merge with pyin for single-note / low-note frames ------------------
   for t in range(min(len(frame_notes), len(f0))):
       if voiced_flag[t] and not np.isnan(f0[t]):
           pyin_midi = int(round(librosa.hz_to_midi(f0[t])))
           if 21 <= pyin_midi <= 108:
               cqt_set = frame_notes[t]
               if len(cqt_set) <= 1:
                   cqt_set.add(pyin_midi)
                   sub = pyin_midi - 12
                   if sub in cqt_set and sub >= 21:
                       cqt_set.discard(pyin_midi)
               else:
                   # Multi-note: if pyin sees a note whose octave-up is
                   # in the CQT set, replace octave-up with pyin pitch.
                   # Exception: when pyin reports bass (< C3), keep both –
                   # the octave-up may be melody (e.g. A3 over A2).
                   octave_up = pyin_midi + 12
                   if (pyin_midi not in cqt_set and octave_up in cqt_set
                           and pyin_midi >= 48):  # skip when pyin says bass
                       cqt_set.discard(octave_up)
                       cqt_set.add(pyin_midi)

   # --- Basic-pitch referee: melody note recovery ----------------------------
   # Only used to recover notes the HF rejected as octave harmonics.
   # basic-pitch decides WHETHER a note exists; CQT decides WHEN (timing).
   # Chords are entirely from our CQT pipeline — basic-pitch never touches them.
   bp_ranges = {}  # default in case basic-pitch fails
   bp_bass_notes = []
   try:
       import sys
       _tf_blocked = 'tensorflow' not in sys.modules
       if _tf_blocked:
           sys.modules['tensorflow'] = None
       from basic_pitch.inference import predict as bp_predict, Model
       from basic_pitch import ICASSP_2022_MODEL_PATH
       if _tf_blocked and sys.modules.get('tensorflow') is None:
           del sys.modules['tensorflow']

       onnx_path = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
       bp_model = Model(str(onnx_path))

       logger.info("Running basic-pitch (ONNX) for melody disambiguation...")
       bp_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
       try:
           sf.write(bp_tmp.name, y, sr)
           bp_tmp.close()
           _, _, bp_note_events = bp_predict(
               bp_tmp.name,
               model_or_model_path=bp_model,
               onset_threshold=0.5,
               frame_threshold=0.3,
               minimum_note_length=50.0,
           )
       finally:
           if os.path.exists(bp_tmp.name):
               os.unlink(bp_tmp.name)

       # Build per-pitch time ranges from basic-pitch (as frame indices)
       bp_ranges = {}  # pitch -> list of (start_frame, end_frame)
       for start_t, end_t, pitch, amp, _ in bp_note_events:
           p = int(pitch)
           sf_bp = int(round(start_t / (hop / sr)))
           ef_bp = int(round(end_t / (hop / sr)))
           bp_ranges.setdefault(p, []).append((sf_bp, ef_bp))
           logger.info(
               f"  [BP] {midi_to_note_name(p)} "
               f"{start_t:.3f}–{end_t:.3f}s (f={sf_bp}–{ef_bp})")

       # Collect unique octave pairs that were ACTUALLY rejected by the HF
       octave_pairs_rejected = set()
       for rej_list in frame_octave_rejections:
           for rejected_midi, keeper_midi in rej_list:
               octave_pairs_rejected.add((rejected_midi, keeper_midi))

       bp_recoveries = 0
       for rejected_midi, keeper_midi in sorted(octave_pairs_rejected):
           if rejected_midi not in bp_ranges:
               logger.info(
                   f"  [BP-REF] {midi_to_note_name(rejected_midi)}<-"
                   f"{midi_to_note_name(keeper_midi)}: "
                   f"NOT in basic-pitch → stay rejected")
               continue

           # Only recover in frames that are BOTH:
           # 1) HF-rejected by our CQT code
           # 2) Within a basic-pitch detected time range for this pitch
           frames_added = 0
           for t in range(len(frame_notes)):
               was_rejected = any(
                   r == rejected_midi and k == keeper_midi
                   for r, k in frame_octave_rejections[t])
               if not was_rejected or rejected_midi in frame_notes[t]:
                   continue
               # Check if this frame falls within any BP range for this pitch
               in_bp_range = any(
                   sf <= t <= ef for sf, ef in bp_ranges[rejected_midi])
               if in_bp_range:
                   frame_notes[t].add(rejected_midi)
                   frames_added += 1

           if frames_added > 0:
               bp_recoveries += 1
               dur_ms = frames_added * (hop / sr) * 1000
               logger.info(
                   f"  [BP-REF] {midi_to_note_name(rejected_midi)} recovered "
                   f"over {midi_to_note_name(keeper_midi)}: "
                   f"{frames_added} frames ({dur_ms:.0f}ms)")
           else:
               logger.info(
                   f"  [BP-REF] {midi_to_note_name(rejected_midi)}<-"
                   f"{midi_to_note_name(keeper_midi)}: "
                   f"in basic-pitch but no overlap with HF-rejected frames")

       logger.info(f"Basic-pitch melody recoveries: {bp_recoveries}")

       # --- BP bass overtone cleanup ------------------------------------------
       # For deep bass notes confirmed by BP, the CQT often anchors on the
       # 2nd harmonic (octave up) as "strongest" and lets the 3rd/5th
       # harmonics through unfiltered.  Remove upper notes that are harmonics
       # of a BP-confirmed bass note AND not confirmed by BP at this
       # specific frame (frame-level validation, not global pitch set).
       BP_BASS_CUTOFF = 60  # MIDI 60 = displayed C3 — bass register
       bp_pitches_set = set(bp_ranges.keys())
       bp_bass_notes = [p for p in bp_pitches_set if p < BP_BASS_CUTOFF]
       overtone_removals = 0
       if bp_bass_notes:
           for t in range(len(frame_notes)):
               to_remove = set()
               for bass_p in bp_bass_notes:
                   in_range = any(
                       sf <= t <= ef for sf, ef in bp_ranges[bass_p])
                   if not in_range:
                       continue
                   for note_p in list(frame_notes[t]):
                       if note_p == bass_p:
                           continue
                       interval = note_p - bass_p
                       if interval in HARMONIC_INTERVALS:
                           note_confirmed_here = any(
                               sf <= t <= ef
                               for sf, ef in bp_ranges.get(note_p, []))
                           if not note_confirmed_here:
                               to_remove.add(note_p)
               if to_remove:
                   frame_notes[t] -= to_remove
                   overtone_removals += len(to_remove)
           if overtone_removals > 0:
               logger.info(
                   f"BP bass overtone cleanup: removed {overtone_removals} "
                   f"overtone instances across frames")

       # --- BP bass note insertion pass ----------------------------------------
       # The HF often rejects bass notes as harmonics of higher notes
       # (e.g. G2 rejected as 3rd harmonic of B4).  For BP-confirmed bass
       # notes, ensure they appear in frame_notes during their active ranges.
       bass_insertions = 0
       for bass_p in bp_bass_notes:
           for sf_bp, ef_bp in bp_ranges[bass_p]:
               for t in range(sf_bp, min(ef_bp + 1, len(frame_notes))):
                   if bass_p not in frame_notes[t] and frame_rms[t] >= noise_floor:
                       frame_notes[t].add(bass_p)
                       bass_insertions += 1
       if bass_insertions > 0:
           logger.info(
               f"BP bass insertion: added {bass_insertions} bass note "
               f"instances across frames")

   except Exception as e:
       logger.warning(f"Basic-pitch referee failed (falling back): {e}")
       import traceback
       logger.warning(traceback.format_exc())

   # --- Track note on/off with debouncing ----------------------------------
   min_on_frames = 3      # ~32 ms at hop=512, sr=48 kHz
   min_off_frames = 10    # ~107 ms – ride through brief spectral dips


   all_pitches = set()
   for fn in frame_notes:
       all_pitches.update(fn)


   logger.info(f"Unique pitches detected across all frames: {len(all_pitches)}")


   note_events = []


   for pitch in sorted(all_pitches):
       active = [pitch in fn for fn in frame_notes]


       in_note = False
       start_frame = 0
       on_count = 0
       off_count = 0


       for t in range(len(active)):
           if active[t]:
               if in_note:
                   off_count = 0
               else:
                   on_count += 1
                   if on_count >= min_on_frames:
                       in_note = True
                       start_frame = t - min_on_frames + 1
                       off_count = 0
           else:
               on_count = 0
               if in_note:
                   off_count += 1
                   if off_count >= min_off_frames:
                       end_frame = t - min_off_frames
                       _emit_note(note_events, pitch, start_frame, end_frame,
                                  cqt_times, C, fmin_midi, num_frames,
                                  global_cqt_max)
                       in_note = False


       if in_note:
           _emit_note(note_events, pitch, start_frame, num_frames - 1,
                      cqt_times, C, fmin_midi, num_frames, global_cqt_max)


   note_events.sort(key=lambda n: (n['start'], n['pitch']))
   logger.info(f"Frame-wise tracking produced {len(note_events)} note events")


   # --- Onset detection (shared by melody detection & re-articulation) -----
   onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
   onset_frames_arr = librosa.onset.onset_detect(
       y=y, sr=sr, onset_envelope=onset_env, hop_length=hop,
       wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1
   )
   onset_times_arr = librosa.frames_to_time(
       onset_frames_arr, sr=sr, hop_length=hop)
   logger.info(f"Onsets detected: {len(onset_times_arr)}")


   # --- Chord-change detection & re-articulation --------------------------
   # A chord change is detected when the frame_notes set changes
   # significantly: notes appear AND/OR disappear.
   logger.info("Detecting chord changes from frame content...")
   chord_change_onsets = {}  # onset_time -> set of gone pitches


   for oi, onset_frame in enumerate(onset_frames_arr):
       if onset_frame >= num_frames:
           continue
       onset_time = float(cqt_times[min(onset_frame, num_frames - 1)])


       pre_f = max(0, onset_frame - 5)
       post_f = min(len(frame_notes) - 1, onset_frame + 5)


       pre_set = frame_notes[pre_f] if pre_f < len(frame_notes) else set()
       post_set = frame_notes[post_f] if post_f < len(frame_notes) else set()


       new_notes = post_set - pre_set
       gone_notes = pre_set - post_set

       if not new_notes or not gone_notes:
           continue

       # A chord change requires a SUSTAINED note departing (not a brief
       # melody note swapping out).  Count how many consecutive frames each
       # gone note was active before this onset.
       MIN_SUSTAIN_FRAMES = 15  # ~160 ms at hop=512/sr=48k
       sustained_departure = False
       for gn in gone_notes:
           consecutive = 0
           for lookback in range(1, int(onset_frame) + 1):
               f_idx = int(onset_frame) - lookback
               if f_idx < 0 or f_idx >= len(frame_notes):
                   break
               if gn in frame_notes[f_idx]:
                   consecutive += 1
               else:
                   break
           if consecutive >= MIN_SUSTAIN_FRAMES:
               sustained_departure = True
               break

       if sustained_departure:
           chord_change_onsets[onset_time] = gone_notes
           logger.info(
               f"  Chord change @{onset_time:.3f}s: "
               f"new={[midi_to_note_name(m) for m in sorted(new_notes)]}, "
               f"gone={[midi_to_note_name(m) for m in sorted(gone_notes)]}")

   BASS_SUSTAIN_CUTOFF = 72  # MIDI 72 = C4 (DAW) — below middle C = bass register
   bass_level_changes = set()

   # Re-articulate: split notes at chord-change onsets.
   # Bass notes are exempt from splits caused by purely melodic changes
   # (where no bass note arrives or departs).  When another bass note
   # is in the new/gone set, all notes — including sustained bass — split.
   if chord_change_onsets:
       for onset_t, gone in chord_change_onsets.items():
           pre_f = max(0, int(round(onset_t / (hop / sr))) - 5)
           post_f = min(len(frame_notes) - 1,
                        int(round(onset_t / (hop / sr))) + 5)
           pre_set = frame_notes[pre_f] if pre_f < len(frame_notes) else set()
           post_set = frame_notes[post_f] if post_f < len(frame_notes) else set()
           new_notes = post_set - pre_set
           has_bass_change = any(
               p < BASS_SUSTAIN_CUTOFF for p in (gone | new_notes))
           if has_bass_change:
               bass_level_changes.add(onset_t)

       new_events = []
       for note in note_events:
           margin = 0.04
           splits = sorted(
               t for t in chord_change_onsets
               if note['start'] + margin < t < note['end'] - margin
               and (note['pitch'] >= BASS_SUSTAIN_CUTOFF
                    or t in bass_level_changes)
           )
           if not splits:
               new_events.append(note)
               continue

           boundaries = [note['start']] + splits + [note['end']]
           for k in range(len(boundaries) - 1):
               seg_s = boundaries[k]
               seg_e = boundaries[k + 1]
               if seg_e - seg_s >= 0.04:
                   new_events.append({
                       'pitch': int(note['pitch']),
                       'start': float(seg_s),
                       'end': float(seg_e),
                       'velocity_raw': note['velocity_raw'],
                   })
       note_events = new_events


   # --- Per-note re-articulation at non-chord-change onsets ----------------
   # A melody note that re-strikes a pitch already in the chord (e.g. A3
   # played as melody while A3 is sustaining in the Am chord) won't create
   # a new note event because the pitch never left the active set.
   # Detect this by looking for an energy spike in the note's CQT bin at
   # each onset.  If the energy dips then rises, split the note there.
   logger.info("Running per-note re-articulation...")
   reartic_count = 0


   for onset_frame in onset_frames_arr:
       if onset_frame >= num_frames:
           continue
       onset_time = float(cqt_times[min(onset_frame, num_frames - 1)])


       # Skip onsets that were already handled as chord changes
       if onset_time in chord_change_onsets:
           continue


       new_events = []
       split_happened = False


       for note in note_events:
           margin = 0.06
           # Only consider notes that span well across this onset
           if not (note['start'] + margin < onset_time < note['end'] - margin):
               new_events.append(note)
               continue


           bin_idx = note['pitch'] - fmin_midi
           if not (0 <= bin_idx < C.shape[0]):
               new_events.append(note)
               continue


           of = int(onset_frame)


           # Compare energy BEFORE the onset vs AT/AFTER the onset
           # Use a gap so the "before" window doesn't overlap the attack
           pre_s = max(0, of - 8)
           pre_e = max(0, of - 2)
           post_s = of
           post_e = min(num_frames, of + 5)


           pre_energy = (float(np.mean(C[bin_idx, pre_s:pre_e]))
                         if pre_e > pre_s else 0.0)
           post_energy = (float(np.mean(C[bin_idx, post_s:post_e]))
                          if post_e > post_s else 0.0)


           # Re-strike: energy rises by at least 20% (attack on top of sustain)
           if pre_energy > 0 and post_energy > pre_energy * 1.20:
               # Split at onset
               if onset_time - note['start'] >= 0.04:
                   new_events.append({
                       'pitch': int(note['pitch']),
                       'start': float(note['start']),
                       'end': float(onset_time),
                       'velocity_raw': note['velocity_raw'],
                   })
               if note['end'] - onset_time >= 0.04:
                   new_events.append({
                       'pitch': int(note['pitch']),
                       'start': float(onset_time),
                       'end': float(note['end']),
                       'velocity_raw': note['velocity_raw'],
                   })
               split_happened = True
               reartic_count += 1
               logger.info(
                   f"  Re-articulation: {midi_to_note_name(note['pitch'])} "
                   f"split @{onset_time:.3f}s "
                   f"(pre={pre_energy:.3f} post={post_energy:.3f})")
           else:
               new_events.append(note)


       if split_happened:
           note_events = new_events


   logger.info(f"Per-note re-articulations: {reartic_count}")

   # --- Direct BP bass event creation ----------------------------------------
   # For bass notes, bypass the CQT tracker entirely and create events
   # directly from BP's segments.  Strategy:
   #   0. Filter BP bass pitches that are octave harmonics of lower BP bass
   #   1. Merge contiguous BP segments (gap ≤ 50ms) — melody-induced splits
   #   2. Real gaps (> 50ms) stay as separate events
   #   3. Split at chord changes involving BP-confirmed bass note movement;
   #      discard residual segments where the pitch itself was "gone"
   BASS_DIRECT_CUTOFF = 72  # MIDI 72 = C4 (DAW)
   CONTIGUOUS_GAP = 0.05    # merge BP segments with gaps ≤ 50ms
   if bp_ranges and bp_bass_notes:
       bp_bass_pitches = set(p for p in bp_ranges if p < BASS_DIRECT_CUTOFF)

       # Step 0: filter octave-harmonic overtones
       OCTAVE_INTERVALS = {12, 24, 36}
       overtone_pitches = set()
       for p in sorted(bp_bass_pitches):
           for lower_p in bp_bass_pitches:
               if lower_p >= p:
                   continue
               if (p - lower_p) not in OCTAVE_INTERVALS:
                   continue
               p_segs = bp_ranges[p]
               lower_segs = bp_ranges[lower_p]
               p_min = min(s for s, _ in p_segs)
               p_max = max(e for _, e in p_segs)
               if any(s <= p_max and e >= p_min for s, e in lower_segs):
                   overtone_pitches.add(p)
                   logger.info(
                       f"  BP bass overtone filter: {midi_to_note_name(p)} "
                       f"is octave of {midi_to_note_name(lower_p)} → skipped")
                   break

       bp_valid_bass = bp_bass_pitches - overtone_pitches

       # Build refined bass chord changes: only changes where a
       # BP-confirmed (non-overtone) bass pitch arrives or departs
       frame_dur = hop / sr
       refined_bass_changes = {}
       for onset_t, gone in chord_change_onsets.items():
           of = max(0, int(round(onset_t / frame_dur)))
           pre_f = max(0, of - 5)
           post_f = min(len(frame_notes) - 1, of + 5)
           pre_set = frame_notes[pre_f] if pre_f < len(frame_notes) else set()
           post_set = frame_notes[post_f] if post_f < len(frame_notes) else set()
           new_notes = post_set - pre_set
           bass_gone = set(
               p for p in gone if p < BASS_DIRECT_CUTOFF and p in bp_valid_bass)
           bass_new = set(
               p for p in new_notes
               if p < BASS_DIRECT_CUTOFF and p in bp_valid_bass)
           if bass_gone or bass_new:
               refined_bass_changes[onset_t] = bass_gone

       bass_change_times = sorted(refined_bass_changes.keys())

       # Build BP-derived bass events
       bp_bass_events = []
       for bass_p in sorted(bp_valid_bass):
           segs = bp_ranges[bass_p]
           bp_times = []
           for sf_bp, ef_bp in segs:
               seg_s = float(cqt_times[min(sf_bp, num_frames - 1)])
               seg_e = float(cqt_times[min(ef_bp, num_frames - 1)])
               if seg_e > seg_s:
                   bp_times.append((seg_s, seg_e))
           if not bp_times:
               continue
           bp_times.sort()

           # Step 1: merge contiguous segments (gap ≤ CONTIGUOUS_GAP)
           merged = [list(bp_times[0])]
           for seg_s, seg_e in bp_times[1:]:
               if seg_s - merged[-1][1] <= CONTIGUOUS_GAP:
                   merged[-1][1] = max(merged[-1][1], seg_e)
               else:
                   merged.append([seg_s, seg_e])

           # Step 2: split at refined bass chord changes, then discard
           # residual tails where this pitch was in the "gone" set
           final_segs = []
           for seg_s, seg_e in merged:
               splits = sorted(
                   t for t in bass_change_times
                   if seg_s + 0.04 < t < seg_e - 0.04)
               if not splits:
                   final_segs.append((seg_s, seg_e))
               else:
                   boundaries = [seg_s] + splits + [seg_e]
                   for k in range(len(boundaries) - 1):
                       s, e = boundaries[k], boundaries[k + 1]
                       if e - s < 0.05:
                           continue
                       if k > 0:
                           gone_at_split = refined_bass_changes.get(
                               boundaries[k], set())
                           if bass_p in gone_at_split:
                               logger.info(
                                   f"  Discarding residual "
                                   f"{midi_to_note_name(bass_p)} after "
                                   f"{boundaries[k]:.3f}s (pitch was 'gone')")
                               continue
                       final_segs.append((s, e))

           for seg_s, seg_e in final_segs:
               bp_bass_events.append({
                   'pitch': int(bass_p),
                   'start': float(seg_s),
                   'end': float(seg_e),
                   'velocity_raw': 70,
               })

       # Diagnostic: log each BP-derived bass event
       for ev in sorted(bp_bass_events, key=lambda e: (e['pitch'], e['start'])):
           note_name = midi_to_note_name(ev['pitch'])
           logger.info(f"  BP bass event: {note_name} "
                       f"{ev['start']:.3f}s – {ev['end']:.3f}s "
                       f"({ev['end'] - ev['start']:.3f}s)")

       # Replace tracker's bass events with BP-derived ones
       tracker_bass = [n for n in note_events if n['pitch'] < BASS_DIRECT_CUTOFF]
       melody_events = [n for n in note_events if n['pitch'] >= BASS_DIRECT_CUTOFF]
       logger.info(f"Replacing {len(tracker_bass)} tracker bass events with "
                   f"{len(bp_bass_events)} BP-derived bass events")
       note_events = melody_events + bp_bass_events

   note_events.sort(key=lambda n: (n['start'], n['pitch']))
   logger.info(f"Final note events: {len(note_events)}")


   return note_events, f0, voiced_flag, voiced_probs, cqt_times

'''
50ms minimum duration one problem when transcribing "Für Elise" – the short grace notes get filtered out.  We could consider lowering this threshold to about 25ms or by making it adaptive based on tempo."
"Für Elise" has plenty of rapid 16th and 32nd notes which we saw were not being transcribed in the output.
'''
def _emit_note(note_events, pitch, start_frame, end_frame, cqt_times, C,
              fmin_midi, num_frames, global_cqt_max, min_duration=0.05):
   """Helper: create a note event if it meets minimum duration and energy."""
   sf = max(start_frame, 0)
   ef = min(end_frame, num_frames - 1)
   s = float(cqt_times[sf])
   e = float(cqt_times[ef])
   if e - s < min_duration:
       return


   bin_idx = pitch - fmin_midi
   if not (0 <= bin_idx < C.shape[0]):
       return


   # Velocity from CQT energy at onset (first 5 frames of note)
   onset_end = min(sf + 5, num_frames)
   onset_energy = float(np.mean([C[bin_idx, tf] for tf in range(sf, onset_end)]))


   # Reject notes whose onset energy is < 5% of global CQT max
   if onset_energy < global_cqt_max * 0.05:
       return
   '''
   A short note that barely survives the duration check above gets fitered out here.
   This 5% energy gate is currently filtering out the grace notes in "Für Elise". We should consider lowering this threshold to 1% or by also making it adaptive based on tempo.
   
   A potential issue that might arise from lowering to 1% is that we might start finding too many false positives in the output. 
   If min_duration is changed, we should leave it as that but instead change this energy gate.
   '''

   '''Tried an approach where only if basic pitch detects it as a note, it can be output. This led to too few notes. 
   Lowering the threshold led to too many notes. I tried using a balance of both and did not yield any significant change. The notes in the 
   test file are also extremely short and there are no sustaining notes so the number of frames for note detection might be something to tinker
   with '''

   

   


   vel_raw = int(np.clip(onset_energy / global_cqt_max * 127, 20, 127))


   note_events.append({
       'pitch': int(pitch),
       'start': s,
       'end': e,
       'velocity_raw': vel_raw,
   })




# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_audio(audio_file):
   """
   Analyse an uploaded audio file using frame-by-frame CQT tracking.


   Each pitch is independently tracked for when it appears and disappears,
   so overlapping notes (sustained chords with melody on top) are handled
   naturally.
   """
   logger.info(f"Starting audio analysis for: {audio_file.name}")


   with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
       tmp_file.write(audio_file.getvalue())
       tmp_path = tmp_file.name


   try:
       file_ext = Path(audio_file.name).suffix.lower()


       # Convert non-WAV formats via pydub
       if file_ext in ('.mp3', '.m4a'):
           # Include guard against pydub not supporting the format (e.g. missing ffmpeg):
           # if AudioSegment.ffmpeg is None or not os.path.isfile(AudioSegment.ffmpeg):
           #    raise RuntimeError(f"Cannot convert {file_ext} to WAV: pydub does not have ffmpeg support.")
           # )
           logger.info(f"Converting {file_ext} to WAV...")
           if file_ext == '.mp3':
               audio_seg = AudioSegment.from_mp3(tmp_path)
           else:
               audio_seg = AudioSegment.from_file(tmp_path, 'm4a')
           wav_path = tmp_path.replace(file_ext, '.wav')
           audio_seg.export(wav_path, format='wav')
           load_path = wav_path
       else:
           load_path = tmp_path
           wav_path = None


       # --- Load audio -------------------------------------------------------
       logger.info("Loading audio with librosa...")
       y, sr = librosa.load(load_path, sr=None, mono=True)
       duration = len(y) / sr
       logger.info(f"Loaded: {len(y)} samples, {sr} Hz, {duration:.2f}s")


       if wav_path and os.path.exists(wav_path):
           os.unlink(wav_path)


       # --- Frame-by-frame note detection ------------------------------------
       notes, f0, voiced_flag, voiced_probs, pyin_times = \
           detect_notes_framewise(y, sr)


       # --- Tempo / beat tracking --------------------------------------------
       logger.info("Detecting tempo...")
       hop = 512
       tempo_result = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
       if hasattr(tempo_result, '__len__') and len(tempo_result) == 2:
           tempo_val, beat_frames = tempo_result
       else:
           tempo_val = tempo_result
           beat_frames = np.array([])
       if hasattr(tempo_val, '__len__'):
           tempo_val = float(tempo_val[0]) if len(tempo_val) > 0 else 120.0
       else:
           tempo_val = float(tempo_val)
       beat_times = (librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
                     if len(beat_frames) > 0 else np.array([]))
       logger.info(f"Tempo: {tempo_val:.1f} BPM, {len(beat_times)} beats")


       # --- Spectral features ------------------------------------------------
       # If we plan to use this for audio files that include anything other than piano, we could implement these features for an instrument detection. 
       spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
       spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]


       # --- Log detected notes -----------------------------------------------
       logger.info("")
       logger.info("=" * 80)
       logger.info("DETECTED NOTE EVENTS:")
       logger.info(f"{'#':<4} {'Pitch':<8} {'Start':<10} {'End':<10} "
                    f"{'Dur(ms)':<10} {'Vel':<6}")
       logger.info("-" * 50)
       for i, n in enumerate(notes[:50]):
           name = midi_to_note_name(n['pitch'])
           dur = (n['end'] - n['start']) * 1000
           logger.info(f"{i+1:<4} {name:<8} {n['start']:<10.3f} {n['end']:<10.3f} "
                       f"{dur:<10.0f} {n['velocity_raw']:<6}")
       if len(notes) > 50:
           logger.info(f"... and {len(notes) - 50} more")
       logger.info("=" * 80)


       # --- Build legacy arrays for compatibility ----------------------------
       frequencies = np.where(voiced_flag & ~np.isnan(f0), f0, 0.0)
       onset_times = np.unique([n['start'] for n in notes])


       return {
           'success': True,
           'audio_data': y,
           'sample_rate': sr,
           'notes': notes,
           'frequencies': frequencies,
           'voiced_flags': voiced_flag.copy(),
           'voiced_probabilities': voiced_probs.copy(),
           'times': pyin_times,
           'onset_times': onset_times,
           'tempo': tempo_val,
           'beat_times': beat_times,
           'duration': float(duration),
           'spectral_centroids': spectral_centroids,
           'spectral_rolloff': spectral_rolloff,
       }


   except Exception as e:
       logger.error(f"Error during audio analysis: {e}")
       logger.error(traceback.format_exc())
       return {'success': False, 'error': str(e)}


   finally:
       if os.path.exists(tmp_path):
           os.unlink(tmp_path)




# ---------------------------------------------------------------------------
# JSON metadata
# ---------------------------------------------------------------------------


def generate_json_metadata(audio_analysis, filename):
   """Generate JSON metadata from the analysis result."""
   notes_data = []
   for note in audio_analysis.get('notes', []):
       notes_data.append({
           'time': float(note['start']),
           'end': float(note['end']),
           'midi_number': int(note['pitch']),
           'note_name': midi_to_note_name(int(note['pitch'])),
       })


   metadata = {
       'file_name': filename,
       'timestamp': datetime.now().isoformat(),
       'audio_properties': {
           'duration': audio_analysis['duration'],
           'sample_rate': int(audio_analysis['sample_rate']),
           'tempo': audio_analysis['tempo'],
       },
       'analysis': {
           'total_notes_detected': len(notes_data),
           'beat_times': [float(t) for t in audio_analysis['beat_times']],
       },
       'notes': notes_data,
   }
   return metadata




# ---------------------------------------------------------------------------
# MIDI generation
# ---------------------------------------------------------------------------


def generate_midi(audio_analysis, filename):
   """Generate a MIDI file from the detected note events."""
   logger.info(f"Generating MIDI for: {filename}")

    # Hardcoding initial_tempo to 120 BPM is simple, but we could also use the detected tempo from the analysis if desired. This way, opening the MIDI in a DAW would reflect the original tempo of the audio.
   midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=960)
   instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano


   notes = audio_analysis.get('notes', [])
   if len(notes) == 0:
       logger.warning("No notes to write!")
       with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp:
           midi.write(tmp.name)
           with open(tmp.name, 'rb') as f:
               midi_bytes = f.read()
           os.unlink(tmp.name)
       return midi_bytes, 0


   # --- Velocity normalisation -----------------------------------------------
   # CQT magnitude at the fundamental bin is unreliable as a velocity proxy:
   # bass fundamentals are inherently weaker than upper partials, creating
   # a huge artificial dynamic spread.  Set all notes to a uniform velocity.
   UNIFORM_VEL = 70
   velocities = [UNIFORM_VEL] * len(notes)
   logger.info(f"All velocities set to {UNIFORM_VEL}")


   # --- Group notes by approximate start time for logging --------------------
   # (MIDI itself stores individual notes; grouping is only for display)
   TOL = 0.02  # 20 ms tolerance for "simultaneous"
   groups = []
   current_group = []
   current_start = -1.0


   for i, note in enumerate(notes):
       if abs(note['start'] - current_start) <= TOL and current_group:
           current_group.append((i, note))
       else:
           if current_group:
               groups.append(current_group)
           current_group = [(i, note)]
           current_start = note['start']
   if current_group:
       groups.append(current_group)


   # --- Log grouped notes ----------------------------------------------------
   logger.info("")
   logger.info("=" * 80)
   logger.info("NOTES WRITTEN TO MIDI FILE:")
   logger.info(f"{'#':<4} {'Type':<8} {'Pitches':<30} {'Start':<10} "
               f"{'End':<10} {'Vel':<6}")
   logger.info("-" * 70)


   for gi, group in enumerate(groups[:50]):
       pitches = [n['pitch'] for _, n in group]
       starts = [n['start'] for _, n in group]
       ends = [n['end'] for _, n in group]
       vel = velocities[group[0][0]]


       if len(pitches) > 1:
           root_name, chord_type = identify_chord(pitches)
           label = (f"{root_name} {chord_type}" if chord_type
                    else ', '.join(midi_to_note_name(p) for p in pitches))
           ntype = "Chord"
       else:
           label = midi_to_note_name(pitches[0])
           ntype = "Note"


       # Show the range of start/end times in the group
       s_min, s_max = min(starts), max(starts)
       e_min, e_max = min(ends), max(ends)
       start_str = f"{s_min:.3f}"
       end_str = f"{e_min:.3f}–{e_max:.3f}" if abs(e_max - e_min) > 0.01 else f"{e_min:.3f}"


       logger.info(f"{gi+1:<4} {ntype:<8} {label:<30} {start_str:<10} "
                   f"{end_str:<10} {vel:<6}")
       if len(pitches) > 1:
           detail = ', '.join(f"{midi_to_note_name(n['pitch'])}({_fmt_dur(n)})"
                              for _, n in group)
           logger.info(f"{'':4} {'':8} ({detail})")


   if len(groups) > 50:
       logger.info(f"... and {len(groups) - 50} more groups")
   logger.info("=" * 80)


   # --- Write individual notes to MIDI ---------------------------------------
   total_midi_notes = 0
   for i, note in enumerate(notes):
       midi_note = pretty_midi.Note(
           velocity=velocities[i],
           pitch=note['pitch'],
           start=note['start'],
           end=note['end'],
       )
       instrument.notes.append(midi_note)
       total_midi_notes += 1


   logger.info(f"Total: {len(groups)} groups -> {total_midi_notes} MIDI notes")


   midi.instruments.append(instrument)


   # --- Write & verify -------------------------------------------------------
   with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp:
       midi.write(tmp.name)


       verify = pretty_midi.PrettyMIDI(tmp.name)
       v_notes = verify.instruments[0].notes if verify.instruments else []
       logger.info("")
       logger.info("MIDI VERIFICATION:")
       logger.info(f"  Notes in file : {len(v_notes)}")
       logger.info(f"  End time      : {verify.get_end_time():.3f}s")
       for j, vn in enumerate(v_notes[:15]):
           nm = midi_to_note_name(vn.pitch)
           logger.info(f"  {j+1:>3}. {nm:<6} {vn.start:.3f}s – {vn.end:.3f}s  "
                       f"vel={vn.velocity}")
       if len(v_notes) > 15:
           logger.info(f"  ... and {len(v_notes) - 15} more")


       with open(tmp.name, 'rb') as f:
           midi_bytes = f.read()
       os.unlink(tmp.name)


   logger.info(f"MIDI generation complete: {total_midi_notes} notes, "
               f"{len(midi_bytes)} bytes")
   return midi_bytes, total_midi_notes




def _fmt_dur(note):
   """Format note duration for logging."""
   dur = (note['end'] - note['start']) * 1000
   return f"{dur:.0f}ms"




# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main():
   st.set_page_config(
       page_title="DAWAgent - Audio to MIDI",
       page_icon="🎵",
       layout="wide",
   )


   st.title("🎵 DAWAgent – Audio to MIDI Transcription")
   st.markdown("Upload audio files and convert them to MIDI with accurate "
               "pitch detection and transcription.")
   st.markdown("---")


   st.subheader("Upload Audio Files")
   uploaded_files = st.file_uploader(
       "Choose .wav, .mp3, or .m4a files",
       type=["wav", "mp3", "m4a"],
       accept_multiple_files=True,
       help="Upload one or more audio files (.wav, .mp3, or .m4a format)",
   )


   if uploaded_files:
       st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")


       for idx, uploaded_file in enumerate(uploaded_files, 1):
           with st.expander(f"📁 {uploaded_file.name}", expanded=True):
               col1, col2 = st.columns([2, 1])


               with col1:
                   st.write(f"**File Name:** {uploaded_file.name}")
                   st.write(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
                   st.write(f"**File Type:** {uploaded_file.type}")
                   st.audio(uploaded_file, format=uploaded_file.type)


               with col2:
                   st.download_button(
                       label="⬇️ Download Original",
                       data=uploaded_file,
                       file_name=uploaded_file.name,
                       mime=uploaded_file.type,
                       key=f"download_orig_{idx}",
                   )


               st.markdown("---")


               if st.button("🎼 Generate MIDI Transcription",
                            key=f"generate_{idx}"):
                   logger.info("=" * 80)
                   logger.info(f"GENERATE clicked for: {uploaded_file.name}")
                   logger.info("=" * 80)


                   with st.spinner("Analysing audio..."):
                       result = analyze_audio(uploaded_file)


                   if result['success']:
                       st.success("Audio analysis complete!")


                       col_a, col_b, col_c = st.columns(3)
                       with col_a:
                           st.metric("Duration", f"{result['duration']:.2f}s")
                       with col_b:
                           st.metric("Tempo", f"{result['tempo']:.1f} BPM")
                       with col_c:
                           st.metric("Sample Rate", f"{result['sample_rate']} Hz")


                       # JSON metadata
                       with st.spinner("Building metadata..."):
                           metadata = generate_json_metadata(
                               result, uploaded_file.name)
                           json_str = json.dumps(metadata, indent=2)
                           st.success(
                               f"Detected {len(metadata['notes'])} note events")
                           st.download_button(
                               label="⬇️ Download JSON Metadata",
                               data=json_str,
                               file_name=(f"{Path(uploaded_file.name).stem}"
                                          f"_metadata.json"),
                               mime="application/json",
                               key=f"download_json_{idx}",
                           )


                       # MIDI
                       with st.spinner("Generating MIDI..."):
                           midi_bytes, note_count = generate_midi(
                               result, uploaded_file.name)
                           st.success(
                               f"MIDI file generated with {note_count} notes")
                           st.download_button(
                               label="⬇️ Download MIDI File",
                               data=midi_bytes,
                               file_name=(f"{Path(uploaded_file.name).stem}"
                                          f".mid"),
                               mime="audio/midi",
                               key=f"download_midi_{idx}",
                           )


                       # Stats
                       st.markdown("### 📊 Note Statistics")
                       detected = result.get('notes', [])
                       if detected:
                           all_names = [midi_to_note_name(n['pitch'])
                                        for n in detected]
                           unique = sorted(set(all_names))
                           col_x, col_y = st.columns(2)
                           with col_x:
                               st.write(f"**Unique Notes:** "
                                        f"{', '.join(unique)}")
                           with col_y:
                               st.write(f"**Total Note Events:** "
                                        f"{len(detected)}")
                   else:
                       st.error(f"Error: {result['error']}")
   else:
       st.info("👆 Upload audio files to get started")


   # Sidebar
   with st.sidebar:
       st.header("ℹ️ About")
       st.markdown("""
       **DAWAgent** transcribes audio files to MIDI.


       **Supported formats:** WAV, MP3, M4A


       **Features:**
       - Frame-by-frame polyphonic note tracking
       - Overlapping note support (chords + melody)
       - Monophonic pitch refinement (pYIN)
       - CQT-based spectral analysis
       - Tempo and beat tracking
       - Velocity detection
       - Chord identification
       - MIDI file generation
       - JSON metadata export


       **How it works:**
       1. Upload your audio file
       2. Click "Generate MIDI Transcription"
       3. Audio is analysed frame-by-frame for all active pitches
       4. Notes are tracked independently with start/end times
       5. Download the MIDI file and open in your DAW
       """)


       st.markdown("---")
       st.markdown("### 📊 Statistics")
       if uploaded_files:
           total_size = sum(f.size for f in uploaded_files)
           st.metric("Total Files", len(uploaded_files))
           st.metric("Total Size", f"{total_size / 1024:.2f} KB")
       else:
           st.write("No files uploaded yet")


if __name__ == "__main__":
   main()
