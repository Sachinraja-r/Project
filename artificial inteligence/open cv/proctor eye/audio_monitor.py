import time
import threading
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

class AudioMonitor:
    def __init__(self, sample_rate=44100, chunk_size=1024, threshold=0.08):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.running = False
        self.last_audio_level = 0.0
        self.audio_alert_count = 0
        self.is_speech_detected = False
        self._thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
        self.last_audio_level = round(float(volume_norm), 4)
        if volume_norm > self.threshold:
            self.is_speech_detected = True
            self.audio_alert_count += 1
        else:
            self.is_speech_detected = False

    def start(self):
        if not HAS_SOUNDDEVICE:
            print("AudioMonitor: sounddevice not available. Skipping mic input.")
            return
            
        self.running = True
        self._thread = threading.Thread(target=self._run_stream, daemon=True)
        self._thread.start()

    def _run_stream(self):
        try:
            with sd.InputStream(callback=self._audio_callback, channels=1, samplerate=self.sample_rate):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"AudioMonitor stream error: {e}")

    def stop(self):
        self.running = False

    def get_status(self):
        return {
            "audio_level": self.last_audio_level,
            "speech_detected": self.is_speech_detected,
            "alert_count": self.audio_alert_count
        }

if __name__ == "__main__":
    monitor = AudioMonitor()
    monitor.start()
    print("Testing AudioMonitor for 3 seconds...")
    time.sleep(3)
    print("Audio status:", monitor.get_status())
    monitor.stop()
