import requests
import whisper
from datetime import datetime
import time
import io
import os
import pytz
import torch
import numpy as np
import ffmpeg
from threading import Thread
from queue import Queue
import gc
from dotenv import load_dotenv
import signal
import sys
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/work/data/transcribe.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class AudioStreamTranscriber:
    def __init__(self):
        # Load configuration from environment variables
        self.feed_id = os.getenv('BROADCASTIFY_FEED_ID')
        self.auth = (
            os.getenv('BROADCASTIFY_USERNAME'),
            os.getenv('BROADCASTIFY_PASSWORD')
        )

        # Construct the full URL using the feed ID
        self.url = f"https://audio.broadcastify.com/{self.feed_id}.mp3"

        if not all([self.feed_id, self.auth[0], self.auth[1]]):
            raise ValueError("Missing required environment variables: BROADCASTIFY_FEED_ID, BROADCASTIFY_USERNAME, or BROADCASTIFY_PASSWORD")

        # Configure torch for optimal performance
        torch._utils._load_global_deps = lambda obj, *args, **kwargs: obj
        torch.set_grad_enabled(False)  # Disable gradient computation
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

        # Setup device and model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get model size from environment variable (default to "medium" if not set)
        model_size = os.getenv('WHISPER_MODEL_SIZE', 'medium').lower()
        valid_models = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large', 'turbo']
        if model_size not in valid_models:
            raise ValueError(f"Invalid WHISPER_MODEL_SIZE. Must be one of: {', '.join(valid_models)}")

        self.model = whisper.load_model(model_size, device=self.device)

        # Setup timezone
        self.timezone = pytz.timezone('America/Chicago')

        # Audio settings
        self.buffer_duration = 10  # seconds
        self.sample_rate = 16000  # Hz (Whisper's expected sample rate)
        self.bytes_per_second = 16 * 1024  # 16KB per second for 128kbps MP3
        self.buffer_size = self.bytes_per_second * self.buffer_duration
        self.chunk = 1024  # Chunk size for reading stream

        # Setup processing queue
        self.audio_queue = Queue(maxsize=3)  # Limit queue size to prevent memory bloat
        self.running = True

        # Create pre-configured ffmpeg input args
        self.ffmpeg_args = {
            'format': 'f32le',
            'acodec': 'pcm_f32le',
            'ac': 1,
            'ar': self.sample_rate
        }

    def get_formatted_time(self):
        return datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S %Z")

    def decode_mp3_to_pcm(self, mp3_data):
        try:
            process = (
                ffmpeg
                .input('pipe:0')
                .output('pipe:1', **self.ffmpeg_args)
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )

            stdout, stderr = process.communicate(input=mp3_data)
            return np.frombuffer(stdout, dtype=np.float32).copy()

        except Exception as e:
            logging.error(f"Decoding error: {e}")
            return None

    def process_audio_chunk(self, audio_data):
        try:
            pcm_data = self.decode_mp3_to_pcm(audio_data)

            if pcm_data is not None and len(pcm_data) > 0:
                # Convert to tensor once instead of letting Whisper do it
                audio_tensor = torch.from_numpy(pcm_data).to(self.device)

                result = self.model.transcribe(
                    audio_tensor,
                    language="en",
                    task="transcribe",
                    initial_prompt="This is police and emergency dispatch radio communication.",
                    no_speech_threshold=0.6  # Adjust this based on your needs
                )

                return result["text"]

            return None

        except Exception as e:
            logging.error(f"Processing error: {e}")
            return None
        finally:
            # Force garbage collection after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def transcription_worker(self):
        """Worker thread for processing audio chunks"""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                transcription = self.process_audio_chunk(audio_data)
                if transcription:
                    self.write_transcription(transcription)
                self.audio_queue.task_done()
            except Exception:
                continue

    def write_transcription(self, transcription):
        if transcription and transcription.strip():
            timestamp = self.get_formatted_time()
            try:
                with open("/work/data/transcribe.txt", "a", encoding='utf-8') as f:
                    log_entry = f"[{timestamp}] {transcription}\n"
                    f.write(log_entry)
                    f.flush()  # Ensure immediate write to disk
                    logging.info(f"Successfully wrote transcription to file")
            except Exception as e:
                logging.error(f"Error writing transcription: {e}")

    def stream_and_transcribe(self):
        try:
            logging.info("Starting streaming and transcription service")
            logging.info(f"Stream URL: {self.url}")
            logging.info(f"Transcriptions will be saved in: {os.path.abspath('/work/data/transcribe.txt')}")

            # Start transcription worker thread
            worker_thread = Thread(target=self.transcription_worker, daemon=True)
            worker_thread.start()

            # Start streaming with authentication and larger timeout
            session = requests.Session()
            response = session.get(
                self.url,
                auth=self.auth,
                stream=True,
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"Failed to connect to stream. Status code: {response.status_code}")

            buffer = bytearray()

            for chunk in response.iter_content(chunk_size=self.chunk):
                if not self.running:
                    break

                if chunk:
                    buffer.extend(chunk)

                    if len(buffer) >= self.buffer_size:
                        # Add to processing queue if there's room
                        try:
                            self.audio_queue.put(bytes(buffer), timeout=1)
                        except Exception:
                            logging.warning("Queue full, skipping chunk")

                        buffer.clear()

        except KeyboardInterrupt:
            logging.info("\nStopping stream...")
        except Exception as e:
            logging.error(f"Streaming error: {e}")
        finally:
            self.running = False
            session.close()
            if 'worker_thread' in locals():
                worker_thread.join(timeout=5)

def signal_handler(signum, frame):
    logging.info("Received signal to terminate. Shutting down...")
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGTSTP, signal_handler)  # Ctrl+Z
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C

    try:
        transcriber = AudioStreamTranscriber()
        transcriber.stream_and_transcribe()
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        logging.error("Please check your environment variables and try again.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
