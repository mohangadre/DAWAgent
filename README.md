


## 🎼 How It Works

1. **Audio Loading**: Loads and converts audio to mono at 22.05 kHz sample rate
2. **Frame Analysis**: Processes audio in overlapping frames with FFT analysis
3. **Pitch Detection**: Uses spectral peak detection with frequency weighting and harmonic filtering
4. **Note Grouping**: Groups consecutive frames with similar pitches into note events
5. **Velocity Calculation**: Analyzes amplitude envelope to determine note velocities
6. **MIDI Generation**: Creates a MIDI file with detected notes, timing, and dynamics
7. **Chord Identification**: Logs recognized chord types (C major, F#dim7, etc.)

## 📊 Detailed Logging

The application provides comprehensive terminal logs showing:
- Detected frequencies and MIDI notes for each chord
- Chord names and component pitches
- Note start times, durations, and velocities
- Comparison between WAV analysis and MIDI output

## ⚙️ Configuration

Key parameters in `app.py`:
- `hop_length`: Frame hop size (default: 512 samples)
- `max_pitches`: Maximum simultaneous pitches to detect (default: 5)
- `fmin`/`fmax`: Frequency detection range (default: 65-2000 Hz)
- `n_fft`: FFT window size with 8x zero-padding for resolution

## 🎹 Testing

The app has been extensively tested with:
- Individual notes (C scale)
- 3-note chords (C major, F major, G major triads)
- 4-note chords (C major 7th, F major 7th, etc.)
- Complex chord progressions



## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - Web application framework
- [librosa](https://librosa.org/) - Audio analysis
- [pretty_midi](https://github.com/craffel/pretty-midi) - MIDI file generation
- [scipy](https://scipy.org/) - FFT and signal processing
- [pydub](https://github.com/jiaaro/pydub) - Audio file handling

## 📝 License

MIT License - feel free to use and modify for your projects!

## 👨‍💻 Author

**Mohan Gadre**
- GitHub: [@mohangadre](https://github.com/mohangadre)

## 🙏 Acknowledgments

Building with extensive testing and iterative refinement to achieve professional-grade audio-to-MIDI transcription accuracy.

---

**Note**: For best results, use clean, dry audio recordings without heavy reverb or effects. The app is currently optimized for piano and keyboard instruments.
