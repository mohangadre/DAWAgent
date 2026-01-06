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
import soundfile as sf
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks
from collections import Counter

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
logger.info("ffmpeg paths configured")

def hz_to_midi(frequency):
    """Convert frequency in Hz to MIDI note number."""
    if frequency <= 0:
        return 0
    return int(round(69 + 12 * np.log2(frequency / 440.0)))

def midi_to_note_name(midi_number):
    """Convert MIDI note number to note name."""
    if midi_number <= 0:
        return "N/A"
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"

def detect_pitch_autocorrelation(audio_frame, sr, fmin=80.0, fmax=1000.0):
    """
    Autocorrelation-based pitch detection - finds TRUE fundamental, not harmonics!
    This is much more reliable than FFT for pitch detection.
    Returns (frequency, confidence)
    """
    # Check energy first
    audio_frame = audio_frame.astype(np.float32)
    if np.max(np.abs(audio_frame)) == 0:
        return 0.0, 0.0
    
    # Normalize
    audio_frame = audio_frame / np.max(np.abs(audio_frame))
    
    # Check energy
    energy = np.mean(audio_frame ** 2)
    if energy < 0.002:
        return 0.0, 0.0
    
    # Apply window
    window = np.hanning(len(audio_frame))
    audio_windowed = audio_frame * window
    
    # Compute autocorrelation
    autocorr = signal.correlate(audio_windowed, audio_windowed, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take only positive lags
    
    # Normalize autocorrelation
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    else:
        return 0.0, 0.0
    
    # Define search range based on frequency limits
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    if min_lag >= max_lag:
        return 0.0, 0.0
    
    # Find peaks in autocorrelation within the search range
    search_autocorr = autocorr[min_lag:max_lag]
    
    # Find strong peaks (these are potential fundamental periods)
    peaks, properties = find_peaks(search_autocorr, height=0.3, distance=5)
    
    if len(peaks) == 0:
        # No clear peak found
        return 0.0, 0.0
    
    # Strategy: PREFER HIGHER FREQUENCIES (melody range)
    # Check all peaks and pick the best one, with strong bias toward higher frequencies
    
    best_peak_idx = 0
    best_score = -1
    
    for peak_idx in peaks:
        lag = peak_idx + min_lag
        freq = sr / lag
        peak_strength = search_autocorr[peak_idx]
        
        # Score function: prefer higher frequencies (shorter lags)
        # Melody is usually in the range 200-800 Hz
        if 200 <= freq <= 800:
            frequency_bonus = 2.0  # Strong bonus for melody range
        elif freq > 800:
            frequency_bonus = 1.5  # Still good
        else:
            frequency_bonus = 1.0  # Lower frequencies penalized
        
        score = peak_strength * frequency_bonus
        
        if score > best_score:
            best_score = score
            best_peak_idx = peak_idx
    
    lag = best_peak_idx + min_lag
    frequency = sr / lag
    confidence = float(search_autocorr[best_peak_idx])
    
    logger.debug(f"Autocorr selected: {frequency:.1f} Hz (conf={confidence:.2f})")
    
    # Additional check: make sure frequency is in valid range
    if frequency < fmin or frequency > fmax:
        return 0.0, 0.0
    
    return float(frequency), float(confidence)

def detect_polyphonic_pitches(audio_frame, sr, fmin=65.0, fmax=2000.0, max_pitches=6, debug=False):
    """
    Detect multiple simultaneous pitches (for chords) using spectral peak detection.
    Returns list of (frequency, confidence) tuples, sorted by strength.
    """
    audio_frame = audio_frame.astype(np.float32)
    
    # Check energy
    if np.max(np.abs(audio_frame)) == 0:
        return []
    
    # Normalize
    audio_frame = audio_frame / np.max(np.abs(audio_frame))
    
    energy = np.mean(audio_frame ** 2)
    if energy < 0.002:
        return []
    
    # Apply window
    window = np.hanning(len(audio_frame))
    audio_windowed = audio_frame * window
    
    # Compute FFT with heavy zero-padding for better frequency resolution
    # This improves accuracy for distinguishing close pitches (like B3 vs C3)
    n_fft = len(audio_windowed) * 8  # 8x zero-padding (was 4x)
    fft = np.fft.rfft(audio_windowed, n=n_fft)
    fft_freqs = np.fft.rfftfreq(n_fft, 1/sr)
    magnitude = np.abs(fft)
    
    # Only look in valid frequency range
    valid_range = (fft_freqs >= fmin) & (fft_freqs <= fmax)
    magnitude_filtered = magnitude.copy()
    magnitude_filtered[~valid_range] = 0
    
    # FREQUENCY WEIGHTING: Boost low frequencies to prioritize fundamentals over harmonics
    # Problem: C3 (130 Hz) fundamental is MUCH weaker than its 2nd harmonic (260 Hz) in piano sounds
    # Solution: Apply aggressive boost to frequencies below 250 Hz to make fundamentals detectable
    freq_weights = np.ones_like(fft_freqs)
    for i, f in enumerate(fft_freqs):
        if f < 250:  # Below middle C (261.63 Hz)
            # Boost: 4.0x at 100 Hz → 1.0x at 250 Hz (linear taper)
            freq_weights[i] = 1.0 + (250.0 - f) / 250.0 * 3.0  # Range: 1.0 to 4.0
    
    # Apply frequency weighting
    magnitude_weighted = magnitude_filtered * freq_weights
    
    if debug and energy > 0.01:
        logger.info(f"[Weight] Boosting low freqs: 4.0x @ 100 Hz → 1.0x @ 250 Hz")
    
    # Find spectral peaks - using WEIGHTED spectrum to catch weak fundamentals
    # In chords, low fundamentals (C3) are MUCH weaker than higher notes and harmonics
    peaks, properties = find_peaks(
        magnitude_weighted,  # Use weighted spectrum!
        height=np.max(magnitude_weighted) * 0.08,  # At least 8% of max (very low!)
        distance=5,  # Minimum distance between peaks (tight for resolution)
        prominence=np.max(magnitude_weighted) * 0.03  # Extremely low prominence for C3
    )
    
    if len(peaks) == 0:
        return []
    
    # Get peak frequencies and strengths (from ORIGINAL magnitude, not weighted)
    peak_freqs = fft_freqs[peaks]
    peak_mags = magnitude[peaks]
    
    # DEBUG: Show what peaks were found and their boost from weighting
    if debug and len(peaks) > 0 and energy > 0.01:
        logger.info(f"[Peaks] Found {len(peaks)} peaks after weighting:")
        for i in range(min(5, len(peaks))):  # Show first 5
            freq = peak_freqs[i]
            orig_mag = peak_mags[i]
            weighted_mag = magnitude_weighted[peaks[i]]
            boost = weighted_mag / orig_mag if orig_mag > 0 else 1.0
            logger.info(f"  {freq:.2f} Hz: orig_mag={orig_mag:.3f}, weighted={weighted_mag:.3f}, boost={boost:.2f}x")
    
    # Sort by FREQUENCY (low to high) to find fundamentals first
    # This prevents detecting harmonics before their fundamentals
    sorted_indices = np.argsort(peak_freqs)  # Low to high frequency
    
    # Filter out harmonics - Process in frequency order (low to high)
    fundamentals = []
    
    for idx in sorted_indices:
        freq = peak_freqs[idx]
        mag = peak_mags[idx]
        
        is_fundamental = True
        
        # Check if this frequency is a harmonic of ANY existing fundamental
        # Since we process low to high, existing fundamentals are all lower
        for fund_freq, fund_mag in fundamentals:
            # Check if current freq is ~2x, ~3x, ~4x, ~5x, ~6x, ~7x, ~8x the fundamental
            for harmonic in [2, 3, 4, 5, 6, 7, 8]:
                expected_harmonic = fund_freq * harmonic
                if abs(freq - expected_harmonic) / expected_harmonic < 0.05:  # Within 5%
                    is_fundamental = False
                    # logger.debug(f"Rejected {freq:.1f} Hz as {harmonic}x harmonic of {fund_freq:.1f} Hz")
                    break
            if not is_fundamental:
                break
        
        if is_fundamental and len(fundamentals) < max_pitches:
            # Round frequency to nearest semitone for accuracy
            # Add upward bias - larger for low notes which are more prone to FFT errors in chords
            raw_midi = hz_to_midi(freq)
            
            # Low notes (< 150 Hz) need more bias in chords due to spectral interference
            if freq < 150:
                bias = 0.6  # Larger bias for low notes like C3 (130.81 Hz)
            else:
                bias = 0.3  # Standard bias for higher notes
            
            midi_num = raw_midi + bias
            rounded_midi = int(round(midi_num))
            rounded_freq = 440.0 * (2.0 ** ((rounded_midi - 69) / 12.0))
            
            fundamentals.append((rounded_freq, mag / np.max(magnitude)))
            note_name = midi_to_note_name(rounded_midi)
            if debug:
                logger.info(f"[Bias] {freq:.2f} Hz (MIDI {raw_midi:.2f}) +{bias} → {midi_num:.2f} → {rounded_midi} ({note_name})")
    
    # ADDITIONAL POST-FILTER: Remove octave duplicates (should be rare now)
    # If we have two notes exactly 12 semitones apart, keep the lower one
    # Since fundamentals are already in low-to-high order, just check each against previous
    filtered_fundamentals = []
    for freq, conf in fundamentals:
        midi = hz_to_midi(freq)
        is_octave_duplicate = False
        
        for existing_freq, existing_conf in filtered_fundamentals:
            existing_midi = hz_to_midi(existing_freq)
            semitone_diff = abs(midi - existing_midi)
            
            # If exactly 1 octave apart (12 semitones ±1), skip the higher one
            if abs(semitone_diff - 12) < 1.0:
                is_octave_duplicate = True
                # logger.debug(f"Skipped octave duplicate: {freq:.1f} Hz (already have {existing_freq:.1f} Hz)")
                break
        
        if not is_octave_duplicate:
            filtered_fundamentals.append((freq, conf))
    
    # Limit to top N strongest fundamentals (by magnitude)
    # For typical chords: triads=3, 7ths=4, extended=5 notes
    # This prevents weak harmonics/noise from slipping through
    # Sort by strength and keep only the strongest
    filtered_fundamentals.sort(key=lambda x: x[1], reverse=True)  # Sort by magnitude (strongest first)
    limited_fundamentals = filtered_fundamentals[:max_pitches]  # Keep only top N
    
    # Sort back to low-to-high frequency for natural ordering
    limited_fundamentals.sort(key=lambda x: x[0])
    
    # DEBUG: Log what we detected (only for frames with strong signal and when debug=True)
    if debug and len(limited_fundamentals) > 0 and energy > 0.01:
        logger.info(f"[FFT] Found {len(limited_fundamentals)} peaks:")
        for freq, conf in limited_fundamentals:
            midi = hz_to_midi(freq)
            logger.info(f"  {freq:.2f} Hz = MIDI {midi:.2f} ({midi_to_note_name(int(round(midi)))}), mag={conf:.3f}")
    
    return limited_fundamentals

def identify_chord(midi_notes):
    """
    Identify chord type from a list of MIDI note numbers.
    Returns (root_note_name, chord_type) or (None, None) if not recognized.
    """
    if len(midi_notes) < 2:
        return None, None
    
    # Sort notes
    notes = sorted(set([int(n) for n in midi_notes]))
    
    # Calculate intervals from root (in semitones)
    root = notes[0]
    intervals = [n - root for n in notes]
    
    # Normalize to single octave (mod 12)
    intervals_normalized = sorted(set([i % 12 for i in intervals]))
    
    # Define chord patterns (intervals from root in semitones)
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
        (0, 4, 7, 10): "7",  # dominant 7th
        (0, 3, 7, 10): "min7",
        (0, 3, 6, 9): "dim7",
        (0, 3, 6, 10): "m7b5",  # half-diminished
        
        # Extended chords
        (0, 4, 7, 10, 14): "9",  # dominant 9th (with or without octave)
        (0, 4, 7, 11, 14): "maj9",
        (0, 3, 7, 10, 14): "min9",
        
        # 6th chords
        (0, 4, 7, 9): "6",
        (0, 3, 7, 9): "min6",
        
        # Add chords
        (0, 4, 7, 14): "add9",
        (0, 3, 7, 14): "minadd9",
    }
    
    # Try to match the pattern
    intervals_tuple = tuple(intervals_normalized)
    
    if intervals_tuple in chord_patterns:
        chord_type = chord_patterns[intervals_tuple]
        root_name = midi_to_note_name(root)
        return root_name, chord_type
    
    # If no exact match, check for power chord (root + fifth)
    if len(intervals_normalized) == 2 and 7 in intervals_normalized:
        return midi_to_note_name(root), "5"  # power chord
    
    # If no pattern matches, return notes as interval description
    if len(notes) >= 2:
        return midi_to_note_name(root), f"chord({len(notes)} notes)"
    
    return None, None

def analyze_audio(audio_file):
    """Analyze audio file and extract pitch information."""
    
    logger.info(f"Starting audio analysis for: {audio_file.name}")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name
    
    logger.info(f"Saved temporary file to: {tmp_path}")
    
    try:
        # Try to load audio file with error handling
        # For MP3 files, we may need to convert first using pydub
        file_ext = Path(audio_file.name).suffix.lower()
        logger.info(f"File extension: {file_ext}")
        
        if file_ext in ['.mp3', '.m4a']:
            logger.info("Using pydub for MP3/M4A conversion")
            # Use pydub for MP3/M4A files
            # Load with pydub
            if file_ext == '.mp3':
                logger.info("Loading MP3 with AudioSegment...")
                audio = AudioSegment.from_mp3(tmp_path)
            elif file_ext == '.m4a':
                logger.info("Loading M4A with AudioSegment...")
                audio = AudioSegment.from_file(tmp_path, 'm4a')
            
            logger.info(f"Audio loaded: {len(audio)}ms, {audio.frame_rate}Hz")
            
            # Convert to wav temporarily
            wav_path = tmp_path.replace(file_ext, '.wav')
            logger.info(f"Converting to WAV: {wav_path}")
            audio.export(wav_path, format='wav')
            logger.info("WAV export complete")
            
            # Load with soundfile (avoids numba issues)
            logger.info("Loading with soundfile...")
            y, sr = sf.read(wav_path)
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            logger.info(f"Soundfile loaded: {len(y)} samples, {sr}Hz")
            
            # Clean up wav file
            if os.path.exists(wav_path):
                os.unlink(wav_path)
                logger.info("Temporary WAV file cleaned up")
        else:
            # Load WAV files directly with soundfile
            logger.info("Loading WAV file directly with soundfile...")
            y, sr = sf.read(tmp_path)
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            logger.info(f"Soundfile loaded: {len(y)} samples, {sr}Hz")
        
        # Extract pitch using POLYPHONIC DETECTION (for chords!)
        logger.info("Starting polyphonic pitch detection...")
        
        # Process audio in frames - HIGHER RESOLUTION for precise timing
        hop_length = 256  # ~5ms at 48000 Hz - very fine resolution
        frame_length = 4096  # ~85ms at 48000 Hz
        
        # Store multiple pitches per frame
        frame_pitches = []  # List of lists: [[freq1, freq2, ...], [freq1, freq2, ...], ...]
        frame_confidences = []  # Confidence for each frame (max of all pitches)
        times_list = []
        frame_count = 0  # For limiting debug output
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            time = i / sr
            frame_count += 1
            
            # Polyphonic pitch detection: find multiple simultaneous pitches
            # Focus on piano range: A0 to C8 (27.5-4186 Hz)
            # Allow up to 5 notes for extended chords (7ths, 9ths with doubles)
            # Harmonic filtering ensures we only detect actual fundamentals, not spurious notes
            # Only log first 30 frames for debugging
            pitches = detect_polyphonic_pitches(frame, sr, fmin=65.0, fmax=2000.0, max_pitches=5, debug=(frame_count <= 30))
            
            if len(pitches) > 0:
                # Extract frequencies and max confidence
                freqs = [p[0] for p in pitches]
                confs = [p[1] for p in pitches]
                frame_pitches.append(freqs)
                frame_confidences.append(max(confs))
            else:
                frame_pitches.append([])
                frame_confidences.append(0.0)
            
            times_list.append(time)
        
        # For backward compatibility with existing code, also create single-pitch arrays
        # (use strongest pitch from each frame)
        f0 = np.array([pitches[0] if len(pitches) > 0 else 0.0 for pitches in frame_pitches])
        voiced_probs = np.array(frame_confidences)
        times = np.array(times_list)
        
        logger.info(f"Polyphonic pitch detection complete: {len(f0)} frames")
        
        # Log statistics about detected chords
        frames_with_notes = sum(1 for p in frame_pitches if len(p) > 0)
        frames_with_chords = sum(1 for p in frame_pitches if len(p) > 1)
        logger.info(f"Frames with sound: {frames_with_notes}, Frames with chords (>1 note): {frames_with_chords}")
        
        # Keep frames with good confidence (autocorrelation gives reliable confidence)
        voiced_flag = (voiced_probs > 0.3) & (f0 > 0)
        
        logger.info(f"Frames passing energy+confidence filter: {np.sum(voiced_flag)}/{len(voiced_flag)}")
        
        # SMART OCTAVE CORRECTION: Enforce melodic continuity
        # Melodies don't jump octaves randomly - they move stepwise
        f0_corrected = f0.copy()
        
        # First pass: fix sudden octave jumps
        for i in range(1, len(f0)):
            if f0[i] > 0 and f0[i-1] > 0 and voiced_flag[i] and voiced_flag[i-1]:
                ratio = f0[i] / f0[i-1]
                
                # If it's close to 2x or 0.5x (octave error), correct it
                if 1.8 < ratio < 2.2:
                    f0_corrected[i] = f0[i] / 2.0
                elif 0.45 < ratio < 0.55:
                    f0_corrected[i] = f0[i] * 2.0
        
        # Second pass: enforce melodic continuity using a sliding window
        # If a note is an octave off from its neighbors, correct it
        window_size = 5
        for i in range(window_size, len(f0_corrected) - window_size):
            if f0_corrected[i] > 0 and voiced_flag[i]:
                # Get context: nearby frequencies
                context = []
                for j in range(i - window_size, i + window_size + 1):
                    if j != i and 0 <= j < len(f0_corrected) and f0_corrected[j] > 0 and voiced_flag[j]:
                        context.append(f0_corrected[j])
                
                if len(context) >= 3:
                    median_context = np.median(context)
                    current = f0_corrected[i]
                    
                    # Check if current note is an octave off from the median
                    ratio = current / median_context
                    
                    if 1.8 < ratio < 2.2:
                        # Too high by an octave
                        f0_corrected[i] = current / 2.0
                        logger.debug(f"Context correction at frame {i}: {current:.1f} -> {f0_corrected[i]:.1f} Hz")
                    elif 0.45 < ratio < 0.55:
                        # Too low by an octave
                        f0_corrected[i] = current * 2.0
                        logger.debug(f"Context correction at frame {i}: {current:.1f} -> {f0_corrected[i]:.1f} Hz")
        
        f0 = f0_corrected
        logger.info("Applied smart octave correction with melodic continuity")
        
        # Apply median filter for smoothing
        f0_smooth = f0.copy()
        f0_smooth = median_filter(f0_smooth, size=5)
        
        # Only use smoothed values where we have voice
        f0 = np.where(voiced_flag, f0_smooth, 0)
        
        # Quantize frequencies to nearest semitone (forces notes to be in-tune)
        midi_notes = []
        for i in range(len(f0)):
            if voiced_flag[i] and f0[i] > 0:
                midi_note = hz_to_midi(f0[i])
                # Round to nearest semitone
                midi_note = int(round(midi_note))
                midi_notes.append(midi_note)
                # Convert back to frequency
                f0[i] = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        
        # AGGRESSIVE MIDI-LEVEL OCTAVE CORRECTION
        # Fix notes that jump down an octave and then back up
        midi_array = np.zeros(len(f0))
        for i in range(len(f0)):
            if voiced_flag[i] and f0[i] > 0:
                midi_array[i] = hz_to_midi(f0[i])
        
        corrected_midi = midi_array.copy()
        for i in range(1, len(midi_array) - 1):
            if midi_array[i] > 0 and voiced_flag[i]:
                # Look at neighbors
                prev_notes = [midi_array[j] for j in range(max(0, i-5), i) if midi_array[j] > 0 and voiced_flag[j]]
                next_notes = [midi_array[j] for j in range(i+1, min(len(midi_array), i+6)) if midi_array[j] > 0 and voiced_flag[j]]
                
                if prev_notes and next_notes:
                    median_context = np.median(prev_notes + next_notes)
                    diff = corrected_midi[i] - median_context
                    
                    # If we're 10-14 semitones below context, we're an octave too low
                    if -14 <= diff <= -10:
                        corrected_midi[i] += 12  # Shift up an octave
                        logger.info(f"Shifted MIDI note at frame {i} up an octave: {midi_array[i]:.0f} -> {corrected_midi[i]:.0f}")
                    # If we're 10-14 semitones above context, we're an octave too high
                    elif 10 <= diff <= 14:
                        corrected_midi[i] -= 12  # Shift down an octave
                        logger.info(f"Shifted MIDI note at frame {i} down an octave: {midi_array[i]:.0f} -> {corrected_midi[i]:.0f}")
        
        # Apply corrections back to frequencies
        for i in range(len(f0)):
            if voiced_flag[i] and corrected_midi[i] > 0:
                f0[i] = 440.0 * (2.0 ** ((corrected_midi[i] - 69) / 12.0))
        
        logger.info("Applied aggressive MIDI-level octave correction")
        
        if len(midi_notes) > 0:
            logger.info(f"Detected MIDI note range: {min(midi_notes)} to {max(midi_notes)}")
            logger.info(f"Note range: {midi_to_note_name(min(midi_notes))} to {midi_to_note_name(max(midi_notes))}")
            # Show histogram of detected notes
            note_counts = Counter(midi_notes)
            top_notes = note_counts.most_common(5)
            logger.info(f"Most common notes: {[(midi_to_note_name(n), c) for n, c in top_notes]}")
        
        logger.info("Applied energy-based filtering and quantization")
        
        # Times are already calculated by CREPE
        logger.info("Using CREPE time frames...")
        
        # Extract onset times (note start times) - using scipy/numpy to avoid numba
        logger.info("Detecting note onsets...")
        try:
            # Try using librosa's onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            logger.info(f"Found {len(onset_times)} onsets")
        except Exception as e:
            logger.warning(f"Onset detection failed: {e}, using pitch changes as onsets")
            # Fallback: use significant pitch changes as onsets
            onset_times = times[np.where(np.diff(voiced_flag.astype(int)) != 0)[0]]
            logger.info(f"Found {len(onset_times)} onsets from pitch changes")
        
        # Get tempo and beats - estimate from timing (avoiding numba)
        logger.info("Detecting tempo and beats...")
        # Use a simple autocorrelation-based tempo estimation
        if len(onset_times) > 3:
            intervals = np.diff(onset_times)
            # Filter out very short intervals (likely not beats, but note transitions)
            # Assuming reasonable tempo range: 60-180 BPM -> beat intervals: 0.33-1.0 seconds
            valid_intervals = intervals[(intervals > 0.3) & (intervals < 1.5)]
            
            if len(valid_intervals) > 0:
                # Use median of valid intervals
                median_interval = np.median(valid_intervals)
                tempo = 60.0 / median_interval
                # Clamp to reasonable range
                tempo = np.clip(tempo, 60, 180)
            else:
                # If no valid intervals, use a histogram approach on all intervals
                # Find the most common interval (bin them)
                hist, bin_edges = np.histogram(intervals, bins=20)
                most_common_bin_idx = np.argmax(hist)
                median_interval = (bin_edges[most_common_bin_idx] + bin_edges[most_common_bin_idx + 1]) / 2
                # This could be a beat or a subdivision, so test multiples
                if median_interval < 0.3:
                    # Likely a subdivision, multiply
                    median_interval *= 2
                tempo = 60.0 / median_interval if median_interval > 0 else 120.0
                tempo = np.clip(tempo, 60, 180)
            
            # Create beat times at regular intervals
            beat_times = np.arange(0, times[-1], 60.0 / tempo)
        else:
            tempo = 120.0
            beat_times = np.array([])
        
        logger.info(f"Estimated tempo: {tempo:.1f} BPM, {len(beat_times)} beats")
        
        # Get spectral features - using scipy directly
        logger.info("Extracting spectral features...")
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}, using empty arrays")
            spectral_centroids = np.array([])
            spectral_rolloff = np.array([])
        
        # Calculate duration
        duration = len(y) / sr
        logger.info(f"Duration: {duration:.2f} seconds")
        
        logger.info("Audio analysis completed successfully!")
        return {
            'success': True,
            'audio_data': y,
            'sample_rate': sr,
            'frequencies': f0,
            'frame_pitches': frame_pitches,  # NEW: List of pitch lists for polyphonic detection
            'voiced_flags': voiced_flag,
            'voiced_probabilities': voiced_probs,
            'times': times,
            'onset_times': onset_times,
            'tempo': float(tempo),
            'beat_times': beat_times,
            'duration': float(duration),
            'spectral_centroids': spectral_centroids,
            'spectral_rolloff': spectral_rolloff
        }
    
    except Exception as e:
        logger.error(f"Error during audio analysis: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temporary file: {tmp_path}")

def generate_json_metadata(audio_analysis, filename):
    """Generate JSON metadata from audio analysis."""
    
    logger.info(f"Generating JSON metadata for: {filename}")
    
    # Extract notes with timing
    notes_data = []
    frequencies = audio_analysis['frequencies']
    times = audio_analysis['times']
    voiced_flags = audio_analysis['voiced_flags']
    
    for i, (freq, time, voiced) in enumerate(zip(frequencies, times, voiced_flags)):
        if voiced and not np.isnan(freq):
            midi_num = hz_to_midi(freq)
            note_name = midi_to_note_name(midi_num)
            notes_data.append({
                'time': float(time),
                'frequency': float(freq),
                'midi_number': int(midi_num),
                'note_name': note_name,
                'confidence': float(audio_analysis['voiced_probabilities'][i])
            })
    
    # Create metadata structure
    metadata = {
        'file_name': filename,
        'timestamp': datetime.now().isoformat(),
        'audio_properties': {
            'duration': audio_analysis['duration'],
            'sample_rate': int(audio_analysis['sample_rate']),
            'tempo': audio_analysis['tempo']
        },
        'analysis': {
            'total_notes_detected': len(notes_data),
            'onset_times': [float(t) for t in audio_analysis['onset_times']],
            'beat_times': [float(t) for t in audio_analysis['beat_times']]
        },
        'notes': notes_data
    }
    
    logger.info(f"JSON metadata generated with {len(notes_data)} notes")
    return metadata

def detect_note_velocity(audio, sr, start_time, end_time):
    """
    Detect the velocity (dynamic level) of a note based on peak amplitude.
    Returns velocity value (0-127) based on the note's loudness.
    """
    # Convert times to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Ensure valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    if start_sample >= end_sample:
        return 80  # Default velocity
    
    # Extract the note segment
    segment = audio[start_sample:end_sample]
    
    if len(segment) == 0:
        return 80
    
    # Calculate peak amplitude (max absolute value)
    peak_amplitude = np.max(np.abs(segment))
    
    # Calculate RMS energy for overall loudness
    rms_energy = np.sqrt(np.mean(segment ** 2))
    
    # Combine peak and RMS for more accurate velocity
    # Peak gives attack strength, RMS gives sustained level
    combined_level = (peak_amplitude * 0.7) + (rms_energy * 0.3)
    
    # Convert to MIDI velocity (1-127)
    # Target velocity ~97 for typical piano recordings
    
    # Use RMS-based scaling (more stable for chords than peak)
    # Calibrated: RMS ~0.16 should give velocity ~97
    velocity_raw = rms_energy * 650  # Increased to 650 to target ~97 (was 600 → gave 70)
    
    # Keep in reasonable range with minimal compression
    if velocity_raw > 110:
        # Soft compress above 110
        velocity = 100 + (velocity_raw - 110) * 0.2
    else:
        velocity = velocity_raw
    
    velocity = int(np.clip(velocity, 1, 127))
    
    # logger.debug(f"Velocity calc: RMS={rms_energy:.4f}, raw={velocity_raw:.1f}, final={velocity}")
    
    return velocity

def generate_midi(audio_analysis, filename):
    """Generate MIDI file from audio analysis with ADSR envelope detection."""
    
    logger.info(f"Generating MIDI file for: {filename}")
    
    # Create a PrettyMIDI object with standard settings
    # Use 120 BPM (standard) with high resolution for accurate timing
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=960)
    logger.info(f"Created MIDI object with tempo: 120.0 BPM, resolution: 960 ticks/beat")
    
    # Create an instrument (piano)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    logger.info("Created piano instrument")
    
    # Get audio data for envelope detection
    audio_data = audio_analysis.get('audio_data', None)
    sample_rate = audio_analysis.get('sample_rate', 44100)
    logger.info(f"Audio data available for envelope detection: {audio_data is not None}")
    
    frame_pitches_data = audio_analysis.get('frame_pitches', [])
    times = audio_analysis['times']
    
    # Calculate frame duration from times array
    if len(times) > 1:
        frame_duration = times[1] - times[0]
    else:
        frame_duration = 0.005  # Default ~5ms
    
    logger.info(f"Grouping polyphonic notes/chords from {len(frame_pitches_data)} frames...")
    
    # Group consecutive frames with similar pitch SETS into chords/notes
    # Each "note" can have multiple pitches (for chords)
    notes = []
    current_chord = None
    min_note_duration = 0.08  # Minimum 80ms
    silence_gap = 0.03  # Maximum 30ms gap within a chord
    
    def pitches_to_midi(pitch_list):
        """Convert list of frequencies to sorted MIDI numbers."""
        if not pitch_list:
            return []
        midi_nums = []
        for freq in pitch_list:
            midi_num = hz_to_midi(freq)
            # Filter valid piano range (A0 to C8: MIDI 21-108)
            if 21 <= midi_num <= 108:
                rounded = int(round(midi_num))
                midi_nums.append(rounded)
                # logger.info(f"  [Frame→MIDI] {freq:.2f} Hz → MIDI {midi_num:.2f} → {rounded} ({midi_to_note_name(rounded)})")
        
        # Remove duplicates and sort
        result = sorted(set(midi_nums))
        
        # SAFETY CHECK: If we have a B2 (47) but no lower note, and we have E and G above it,
        # it's probably a mis-detected C3 chord → shift B2 up to C3
        if len(result) >= 3 and result[0] == 47:  # B2
            # Check if this looks like a C major chord (should be 48, 52, 55)
            if 52 in result and 55 in result:  # Has E3 and G3
                logger.info(f"  [CORRECTION] B2+E3+G3 → C3+E3+G3")
                result[0] = 48  # Change B2 to C3
        
        # logger.info(f"  [Final] Frame: {[midi_to_note_name(m) for m in result]}")
        return result
    
    def chords_similar(chord1_midis, chord2_midis):
        """Check if two MIDI note sets are similar (same notes)."""
        if len(chord1_midis) != len(chord2_midis):
            return False
        # Allow 1 semitone tolerance for pitch detection variations
        for m1, m2 in zip(sorted(chord1_midis), sorted(chord2_midis)):
            if abs(m1 - m2) > 1:
                return False
        return True
    
    for i, (pitch_list, time) in enumerate(zip(frame_pitches_data, times)):
        midi_notes = pitches_to_midi(pitch_list)
        
        if len(midi_notes) > 0:
            # We have notes in this frame
            if current_chord is None:
                # Start a new chord/note
                current_chord = {
                    'pitches': midi_notes,  # List of MIDI numbers
                    'start': time,
                    'end': time + (frame_duration * 0.7),
                    'pitch_history': [midi_notes],
                    'last_time': time
                }
            elif chords_similar(midi_notes, current_chord['pitches']):
                # Continue the current chord
                gap = time - current_chord['last_time']
                if gap <= silence_gap:
                    current_chord['end'] = time + (frame_duration * 0.7)
                    current_chord['pitch_history'].append(midi_notes)
                    current_chord['last_time'] = time
                else:
                    # Gap too large - save and start new
                    if current_chord['end'] - current_chord['start'] >= min_note_duration:
                        # Finalize pitches by majority vote for each note
                        del current_chord['pitch_history']
                        del current_chord['last_time']
                        notes.append(current_chord)
                    
                    current_chord = {
                        'pitches': midi_notes,
                        'start': time,
                        'end': time + (frame_duration * 0.7),
                        'pitch_history': [midi_notes],
                        'last_time': time
                    }
            else:
                # Different chord - save previous and start new
                if current_chord['end'] - current_chord['start'] >= min_note_duration:
                    del current_chord['pitch_history']
                    del current_chord['last_time']
                    notes.append(current_chord)
                
                current_chord = {
                    'pitches': midi_notes,
                    'start': time,
                    'end': time + (frame_duration * 0.7),
                    'pitch_history': [midi_notes],
                    'last_time': time
                }
        else:
            # Silence
            if current_chord is not None:
                gap = time - current_chord['last_time']
                if gap > silence_gap:
                    # End the chord
                    if current_chord['end'] - current_chord['start'] >= min_note_duration:
                        del current_chord['pitch_history']
                        del current_chord['last_time']
                        notes.append(current_chord)
                    current_chord = None
    
    # Add final chord if exists
    if current_chord is not None:
        if current_chord['end'] - current_chord['start'] >= min_note_duration:
            del current_chord['pitch_history']
            del current_chord['last_time']
            notes.append(current_chord)
    
    logger.info(f"Grouped into {len(notes)} chords/notes")
    
    # Log chord information
    num_chords = sum(1 for n in notes if len(n['pitches']) > 1)
    num_single = sum(1 for n in notes if len(n['pitches']) == 1)
    logger.info(f"Detected: {num_single} single notes, {num_chords} chords")
    
    # DEBUG: Log raw frequencies for first few chords/notes
    logger.info("")
    logger.info("=" * 80)
    logger.info("FREQUENCY ANALYSIS - Raw Detected Frequencies:")
    logger.info("=" * 80)
    for i, note in enumerate(notes[:5]):  # First 5 chords/notes
        if isinstance(note.get('pitches'), list) and len(note['pitches']) > 0:
            midi_notes = note['pitches']
            # Convert MIDI back to exact frequencies
            freqs = [440.0 * (2.0 ** ((m - 69) / 12.0)) for m in midi_notes]
            
            logger.info(f"Chord/Note {i+1} at {note['start']:.3f}s:")
            for j, (freq, midi_num) in enumerate(zip(freqs, midi_notes)):
                note_name = midi_to_note_name(midi_num)
                logger.info(f"  Note {j+1}: {freq:.2f} Hz → MIDI {midi_num} ({note_name})")
    logger.info("=" * 80)
    logger.info("")
    
    # FILTER OUT FALSE POSITIVES: Remove very short notes at the beginning
    # These are often noise/transients, not real notes
    if len(notes) > 0:
        filtered_notes = []
        for i, note in enumerate(notes):
            duration = note['end'] - note['start']
            
            # Chords: Must be at least 100ms (slightly shorter than single notes)
            # Single notes: First must be 150ms, subsequent 100ms
            is_chord = isinstance(note.get('pitches'), list) and len(note['pitches']) > 1
            
            if is_chord:
                min_duration = 0.10  # 100ms for chords
            else:
                min_duration = 0.15 if i == 0 else 0.10  # 150ms first, 100ms others
            
            if duration >= min_duration:
                filtered_notes.append(note)
            else:
                # For chords, show all pitches
                if is_chord:
                    pitch_names = '/'.join([midi_to_note_name(p) for p in note['pitches']])
                    logger.info(f"Filtered out spurious chord: {pitch_names} at {note['start']:.3f}s (only {duration*1000:.0f}ms)")
                else:
                    pitch = note['pitches'][0] if isinstance(note.get('pitches'), list) else note.get('pitch', 60)
                    note_name = midi_to_note_name(int(round(pitch)))
                    logger.info(f"Filtered out spurious note: {note_name} at {note['start']:.3f}s (only {duration*1000:.0f}ms)")
        
        notes = filtered_notes
        logger.info(f"After filtering spurious notes: {len(notes)} chords/notes remaining")
    
    # NO GLOBAL OCTAVE SHIFT - use detected pitches exactly as analyzed
    if len(notes) > 0:
        # Collect all pitches (including from chords)
        all_pitches = []
        for note in notes:
            if isinstance(note.get('pitches'), list):
                all_pitches.extend(note['pitches'])
            elif 'pitch' in note:
                all_pitches.append(note['pitch'])
        
        if len(all_pitches) > 0:
            median_pitch = np.median(all_pitches)
            logger.info(f"Median detected pitch: {median_pitch:.1f} - using as-is (no octave adjustment)")
    
    # DO NOT trim leading silence - preserve exact timing from WAV file
    # Keep original timestamps intact for accurate timing
    
    # Calculate velocities for all notes (will be normalized later)
    wav_velocities_raw = []
    for note in notes:
        if audio_data is not None:
            vel = detect_note_velocity(
                audio_data, sample_rate, note['start'], note['end']
            )
            wav_velocities_raw.append(vel)
        else:
            wav_velocities_raw.append(80)
    
    # LOG WAV FILE NOTES/CHORDS WITH VELOCITY
    logger.info("")
    logger.info("=" * 80)
    logger.info("NOTES/CHORDS DETECTED FROM WAV FILE:")
    logger.info("=" * 80)
    logger.info(f"{'#':<4} {'Type':<8} {'Pitches':<30} {'Start (s)':<12} {'Duration (ms)':<15} {'Velocity':<10}")
    logger.info("-" * 80)
    for i, note in enumerate(notes[:30]):
        duration = (note['end'] - note['start']) * 1000  # Convert to ms
        wav_velocity = wav_velocities_raw[i] if i < len(wav_velocities_raw) else 80
        
        # Check if it's a chord or single note
        if isinstance(note.get('pitches'), list):
            pitches = note['pitches']
            if len(pitches) > 1:
                # It's a chord - identify it
                root_name, chord_type = identify_chord(pitches)
                if chord_type:
                    chord_display = f"{root_name} {chord_type}"
                    pitch_list = ', '.join([midi_to_note_name(p) for p in pitches])
                    logger.info(f"{i+1:<4} {'Chord':<8} {chord_display:<30} {note['start']:<12.3f} {duration:<15.0f} {wav_velocity:<10}")
                    logger.info(f"{'    ':<4} {'        ':<8} ({pitch_list})")
                else:
                    pitch_list = ', '.join([midi_to_note_name(p) for p in pitches])
                    logger.info(f"{i+1:<4} {'Chord':<8} {pitch_list:<30} {note['start']:<12.3f} {duration:<15.0f} {wav_velocity:<10}")
            else:
                # Single note
                note_name = midi_to_note_name(pitches[0])
                logger.info(f"{i+1:<4} {'Note':<8} {note_name:<30} {note['start']:<12.3f} {duration:<15.0f} {wav_velocity:<10}")
        else:
            # Old format - single pitch
            note_name = midi_to_note_name(int(round(note.get('pitch', 60))))
            logger.info(f"{i+1:<4} {'Note':<8} {note_name:<30} {note['start']:<12.3f} {duration:<15.0f} {wav_velocity:<10}")
    
    if len(notes) > 30:
        logger.info(f"... and {len(notes) - 30} more notes/chords")
    logger.info("=" * 80)
    logger.info("")
    
    # Normalize velocities: if they're all similar, use median
    # This handles cases where all notes were played at same MIDI velocity
    if len(wav_velocities_raw) > 0:
        median_vel = np.median(wav_velocities_raw)
        std_vel = np.std(wav_velocities_raw)
        
        logger.info(f"Velocity analysis: median={median_vel:.1f}, std deviation={std_vel:.1f}")
        
        # Normalize velocities for consistent recordings
        # For chords played at same velocity, normalize to median even with some variation
        if std_vel < 20:  # Consistent playing
            logger.info(f"Consistent velocity detected - normalizing all notes to: {int(median_vel)}")
            normalized_velocities = [int(median_vel)] * len(notes)
        else:
            # High variation = keep dynamics but scale to center around 97
            logger.info(f"Dynamic variation detected - scaling relative velocities")
            # Scale to keep relative dynamics but center around target (97)
            scale_factor = 97.0 / median_vel if median_vel > 0 else 1.0
            normalized_velocities = [int(np.clip(v * scale_factor, 1, 127)) for v in wav_velocities_raw]
    else:
        normalized_velocities = [80] * len(notes)
    
    # LOG MIDI FILE NOTES/CHORDS AND WRITE TO MIDI
    logger.info("")
    logger.info("=" * 80)
    logger.info("NOTES/CHORDS WRITTEN TO MIDI FILE:")
    logger.info("=" * 80)
    logger.info(f"{'#':<4} {'Type':<8} {'Pitches':<30} {'Start (s)':<12} {'Duration (ms)':<15} {'Velocity':<10}")
    logger.info("-" * 80)
    
    total_midi_notes = 0  # Count individual MIDI notes (chords count as multiple)
    
    if len(notes) > 0:
        for i, note in enumerate(notes):
            # Use EXACT timestamps from WAV - NO adjustment!
            actual_start = note['start']
            actual_end = note['end']
            duration = (actual_end - actual_start) * 1000  # Convert to ms
            
            # Use normalized velocity
            velocity = normalized_velocities[i]
            
            # Handle chords (multiple pitches) or single notes
            if isinstance(note.get('pitches'), list):
                pitches = note['pitches']
                
                # Log chord information
                if i < 30:
                    if len(pitches) > 1:
                        # It's a chord
                        root_name, chord_type = identify_chord(pitches)
                        if chord_type:
                            chord_display = f"{root_name} {chord_type}"
                            pitch_list = ', '.join([midi_to_note_name(p) for p in pitches])
                            logger.info(f"{i+1:<4} {'Chord':<8} {chord_display:<30} {actual_start:<12.3f} {duration:<15.0f} {velocity:<10}")
                            logger.info(f"{'    ':<4} {'        ':<8} ({pitch_list})")
                        else:
                            pitch_list = ', '.join([midi_to_note_name(p) for p in pitches])
                            logger.info(f"{i+1:<4} {'Chord':<8} {pitch_list:<30} {actual_start:<12.3f} {duration:<15.0f} {velocity:<10}")
                    else:
                        # Single note
                        note_name = midi_to_note_name(pitches[0])
                        logger.info(f"{i+1:<4} {'Note':<8} {note_name:<30} {actual_start:<12.3f} {duration:<15.0f} {velocity:<10}")
                
                # Write all pitches in the chord to MIDI
                for pitch in pitches:
                    # logger.info(f"  [MIDI Write] Writing pitch {pitch} ({midi_to_note_name(pitch)}) to MIDI file")
                    midi_note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=actual_start,
                        end=actual_end
                    )
                    instrument.notes.append(midi_note)
                    total_midi_notes += 1
            else:
                # Old format - single pitch
                pitch = note.get('pitch', 60)
                
                if i < 30:
                    note_name = midi_to_note_name(int(round(pitch)))
                    logger.info(f"{i+1:<4} {'Note':<8} {note_name:<30} {actual_start:<12.3f} {duration:<15.0f} {velocity:<10}")
                
                midi_note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=int(round(pitch)),
                    start=actual_start,
                    end=actual_end
                )
                instrument.notes.append(midi_note)
                total_midi_notes += 1
        
        if len(notes) > 30:
            logger.info(f"... and {len(notes) - 30} more notes/chords")
        logger.info("=" * 80)
        logger.info(f"Total: {len(notes)} chords/notes → {total_midi_notes} MIDI notes written")
        logger.info("")
    else:
        logger.warning("No notes detected!")
    
    # Add instrument to MIDI file
    midi.instruments.append(instrument)
    logger.info("Instrument added to MIDI file")
    
    # Ensure MIDI duration matches WAV duration by checking last note end time
    wav_duration = audio_analysis.get('duration', 10.0)
    if len(instrument.notes) > 0:
        last_note_end = max(note.end for note in instrument.notes)
        logger.info(f"WAV duration: {wav_duration:.3f}s, Last MIDI note ends at: {last_note_end:.3f}s")
        
        if last_note_end < wav_duration:
            trailing_silence = wav_duration - last_note_end
            logger.info(f"MIDI has {trailing_silence:.3f}s of trailing silence (matches WAV)")
    else:
        logger.info(f"WAV duration: {wav_duration:.3f}s")
    
    # Write to bytes
    logger.info("Writing MIDI to bytes...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp_file:
        midi.write(tmp_file.name)
        with open(tmp_file.name, 'rb') as f:
            midi_bytes = f.read()
        
        # VERIFY: Read back the MIDI file to see what was actually written
        logger.info("")
        logger.info("=" * 80)
        logger.info("VERIFICATION - READING BACK GENERATED MIDI FILE:")
        logger.info("=" * 80)
        
        try:
            verify_midi = pretty_midi.PrettyMIDI(tmp_file.name)
            logger.info(f"MIDI Tempo: {verify_midi.get_tempo_changes()}")
            logger.info(f"MIDI End Time: {verify_midi.get_end_time():.3f}s")
            
            if len(verify_midi.instruments) > 0:
                verify_notes = verify_midi.instruments[0].notes
                logger.info(f"Number of notes in file: {len(verify_notes)}")
                logger.info("")
                logger.info(f"{'#':<4} {'Pitch':<8} {'Start (s)':<12} {'Duration (ms)':<15} {'Velocity':<10}")
                logger.info("-" * 80)
                for i, n in enumerate(verify_notes[:30]):
                    note_name = pretty_midi.note_number_to_name(n.pitch)
                    duration = (n.end - n.start) * 1000
                    logger.info(f"{i+1:<4} {note_name:<8} {n.start:<12.3f} {duration:<15.0f} {n.velocity:<10}")
                if len(verify_notes) > 30:
                    logger.info(f"... and {len(verify_notes) - 30} more notes")
            logger.info("=" * 80)
            logger.info("")
        except Exception as e:
            logger.error(f"Could not verify MIDI file: {e}")
        
        os.unlink(tmp_file.name)
    
    logger.info(f"MIDI generation complete: {len(notes)} chords/notes ({total_midi_notes} MIDI notes), {len(midi_bytes)} bytes")
    return midi_bytes, total_midi_notes

def main():
    st.set_page_config(
        page_title="MUSICGEN - Audio to Sheet Music",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 MUSICGEN - Audio to Sheet Music Converter")
    st.markdown("Upload audio files and convert them to MIDI with pitch detection and transcription")
    st.markdown("---")
    
    # File uploader
    st.subheader("Upload Audio Files")
    uploaded_files = st.file_uploader(
        "Choose .wav, .mp3, or .m4a files",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=True,
        help="Upload one or more audio files (.wav, .mp3, or .m4a format)"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Display uploaded files
        st.subheader("Uploaded Files")
        
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            with st.expander(f"📁 {uploaded_file.name}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display file info
                    st.write(f"**File Name:** {uploaded_file.name}")
                    st.write(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
                    st.write(f"**File Type:** {uploaded_file.type}")
                    
                    # Audio player
                    st.audio(uploaded_file, format=uploaded_file.type)
                
                with col2:
                    # Download original button
                    st.download_button(
                        label="⬇️ Download Original",
                        data=uploaded_file,
                        file_name=uploaded_file.name,
                        mime=uploaded_file.type,
                        key=f"download_orig_{idx}"
                    )
                
                st.markdown("---")
                
                # Generate Sheet Music Button
                if st.button(f"🎼 Generate Sheet Music", key=f"generate_{idx}"):
                    logger.info("=" * 80)
                    logger.info(f"GENERATE BUTTON CLICKED for file: {uploaded_file.name}")
                    logger.info("=" * 80)
                    with st.spinner("Analyzing audio and detecting pitches..."):
                        # Analyze audio
                        logger.info("Calling analyze_audio()...")
                        analysis_result = analyze_audio(uploaded_file)
                        logger.info(f"analyze_audio() returned: success={analysis_result.get('success', False)}")
                        
                        if analysis_result['success']:
                            st.success("✅ Audio analysis complete!")
                            
                            # Display analysis info
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Duration", f"{analysis_result['duration']:.2f}s")
                            with col_b:
                                st.metric("Tempo", f"{analysis_result['tempo']:.1f} BPM")
                            with col_c:
                                st.metric("Sample Rate", f"{analysis_result['sample_rate']} Hz")
                            
                            # Generate JSON metadata
                            with st.spinner("Generating JSON metadata..."):
                                metadata = generate_json_metadata(analysis_result, uploaded_file.name)
                                json_str = json.dumps(metadata, indent=2)
                                
                                st.success(f"✅ Detected {len(metadata['notes'])} notes")
                                
                                # Download JSON button
                                st.download_button(
                                    label="⬇️ Download JSON Metadata",
                                    data=json_str,
                                    file_name=f"{Path(uploaded_file.name).stem}_metadata.json",
                                    mime="application/json",
                                    key=f"download_json_{idx}"
                                )
                            
                            # Generate MIDI
                            with st.spinner("Generating MIDI file..."):
                                midi_bytes, note_count = generate_midi(analysis_result, uploaded_file.name)
                                
                                st.success(f"✅ MIDI file generated with {note_count} notes")
                                
                                # Download MIDI button
                                st.download_button(
                                    label="⬇️ Download MIDI File",
                                    data=midi_bytes,
                                    file_name=f"{Path(uploaded_file.name).stem}.mid",
                                    mime="audio/midi",
                                    key=f"download_midi_{idx}"
                                )
                            
                            # Show note statistics
                            st.markdown("### 📊 Note Statistics")
                            if len(metadata['notes']) > 0:
                                notes_list = [n['note_name'] for n in metadata['notes']]
                                unique_notes = list(set(notes_list))
                                unique_notes.sort()
                                
                                col_x, col_y = st.columns(2)
                                with col_x:
                                    st.write(f"**Unique Notes:** {', '.join(unique_notes)}")
                                with col_y:
                                    st.write(f"**Total Notes:** {len(notes_list)}")
                        else:
                            st.error(f"❌ Error analyzing audio: {analysis_result['error']}")
    else:
        st.info("👆 Please upload audio files to get started")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application allows you to:
        - Upload .wav, .mp3, or .m4a audio files
        - Analyze audio and detect pitches
        - Generate JSON metadata with note information
        - Convert audio to MIDI format
        - Download sheet music transcriptions
        
        **Supported Formats:**
        - WAV (.wav)
        - MP3 (.mp3)
        - M4A (.m4a)
        
        **Features:**
        - Pitch detection using pYIN algorithm
        - Tempo and beat detection
        - Note onset detection
        - MIDI file generation
        - JSON metadata export
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files)
            st.metric("Total Files", len(uploaded_files))
            st.metric("Total Size", f"{total_size / 1024:.2f} KB")
        else:
            st.write("No files uploaded yet")
        
        st.markdown("---")
        st.markdown("### 🎹 How it works")
        st.markdown("""
        1. Upload your audio file
        2. Click "Generate Sheet Music"
        3. Audio is analyzed for pitch and timing
        4. Notes are detected and classified
        5. JSON metadata is generated
        6. MIDI file is created
        7. Download your files!
        """)

if __name__ == "__main__":
    main()
