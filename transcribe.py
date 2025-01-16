"""
Audio Stream Transcriber
-----------------------
A service that streams audio from Broadcastify, processes it using OpenAI's Whisper model,
and saves transcriptions to a file.

Requirements:
- See requirements.txt for dependencies
- Environment variables:
    BROADCASTIFY_FEED_ID: ID of the Broadcastify feed
    BROADCASTIFY_USERNAME: Broadcastify username
    BROADCASTIFY_PASSWORD: Broadcastify password
    WHISPER_MODEL_SIZE: Size of Whisper model (optional, defaults to 'medium')
"""

import io
import gc
import logging
import os
import signal
import sys
from datetime import datetime
from queue import Queue
from threading import Thread

import ffmpeg
import numpy as np
import pytz
import requests
import torch
import whisper
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when there's an issue with the configuration settings."""
    pass


class AudioStreamTranscriber:
    """Handles streaming audio from Broadcastify and transcribing it using Whisper."""
    
    # Constants
    BUFFER_DURATION = 10  # seconds
    SAMPLE_RATE = 16000  # Hz (Whisper's expected sample rate)
    BYTES_PER_SECOND = 16 * 1024  # 16KB per second for 128kbps MP3
    CHUNK_SIZE = 1024  # Chunk size for reading stream
    QUEUE_SIZE = 3  # Maximum size of processing queue
    # Default paths that can be overridden by environment variables
    OUTPUT_PATH = os.getenv('OUTPUT_PATH', os.path.join(os.getcwd(), 'data', 'transcribe.txt'))
    LOG_PATH = os.getenv('LOG_PATH', os.path.join(os.getcwd(), 'data', 'transcribe.log'))

    def __init__(self):
        """Initialize the transcriber with configuration from environment variables."""
        self._load_config()
        self._setup_torch()
        self._setup_model()
        self._setup_processing()

    def _load_config(self):
        """Load and validate configuration from environment variables."""
        load_dotenv()
        
        # Required environment variables
        self.feed_id = os.getenv('BROADCASTIFY_FEED_ID')
        self.username = os.getenv('BROADCASTIFY_USERNAME')
        self.password = os.getenv('BROADCASTIFY_PASSWORD')
        
        if not all([self.feed_id, self.username, self.password]):
            raise ConfigurationError(
                "Missing required environment variables: "
                "BROADCASTIFY_FEED_ID, BROADCASTIFY_USERNAME, or BROADCASTIFY_PASSWORD"
            )
        
        self.url = f"https://audio.broadcastify.com/{self.feed_id}.mp3"
        self.timezone = pytz.timezone('America/Chicago')
        self.buffer_size = self.BYTES_PER_SECOND * self.BUFFER_DURATION

    def _setup_torch(self):
        """Configure PyTorch settings."""
        torch._utils._load_global_deps = lambda obj, *args, **kwargs: obj
        torch.set_grad_enabled(False)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def _setup_model(self):
        """Load and configure the Whisper model."""
        model_size = os.getenv('WHISPER_MODEL_SIZE', 'medium').lower()
        valid_models = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 
                       'small.en', 'medium', 'medium.en', 'large', 'turbo']
        
        if model_size not in valid_models:
            raise ConfigurationError(
                f"Invalid WHISPER_MODEL_SIZE. Must be one of: {', '.join(valid_models)}"
            )
        
        self.model = whisper.load_model(model_size, device=self.device)

    def _setup_processing(self):
        """Set up audio processing configuration."""
        self.audio_queue = Queue(maxsize=self.QUEUE_SIZE)
        self.running = True
        
        self.ffmpeg_args = {
            'format': 'f32le',
            'acodec': 'pcm_f32le',
            'ac': 1,
            'ar': self.SAMPLE_RATE
        }

    def get_formatted_time(self):
        """Get current time formatted with timezone."""
        return datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S %Z")

    def decode_mp3_to_pcm(self, mp3_data):
        """Convert MP3 data to PCM format using ffmpeg."""
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
        """Process a chunk of audio data through the Whisper model."""
        try:
            pcm_data = self.decode_mp3_to_pcm(audio_data)
            if pcm_data is None or len(pcm_data) == 0:
                return None

            audio_tensor = torch.from_numpy(pcm_data).to(self.device)
            result = self.model.transcribe(
                audio_tensor,
                language="en",
                task="transcribe",
                initial_prompt="This is police and emergency dispatch radio communication.",
                no_speech_threshold=0.6
            )

            return result["text"]

        except Exception as e:
            logging.error(f"Processing error: {e}")
            return None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def transcription_worker(self):
        """Worker thread for processing audio chunks from the queue."""
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
        """Write transcription to file with timestamp."""
        if not transcription or not transcription.strip():
            return

        timestamp = self.get_formatted_time()
        try:
            with open(self.OUTPUT_PATH, "a", encoding='utf-8') as f:
                log_entry = f"[{timestamp}] {transcription}\n"
                f.write(log_entry)
                f.flush()
            logging.info("Successfully wrote transcription to file")
        except Exception as e:
            logging.error(f"Error writing transcription: {e}")

    def stream_and_transcribe(self):
        """Main method to stream audio and process it for transcription."""
        try:
            logging.info("Starting streaming and transcription service")
            logging.info(f"Stream URL: {self.url}")
            logging.info(f"Transcriptions will be saved in: {os.path.abspath(self.OUTPUT_PATH)}")

            worker_thread = Thread(target=self.transcription_worker, daemon=True)
            worker_thread.start()

            session = requests.Session()
            response = session.get(
                self.url,
                auth=(self.username, self.password),
                stream=True,
                timeout=30
            )

            if response.status_code != 200:
                raise ConnectionError(f"Failed to connect to stream. Status code: {response.status_code}")

            self._process_stream(response)

        except KeyboardInterrupt:
            logging.info("\nStopping stream...")
        except Exception as e:
            logging.error(f"Streaming error: {e}")
        finally:
            self._cleanup(session, worker_thread if 'worker_thread' in locals() else None)

    def _process_stream(self, response):
        """Process the audio stream in chunks."""
        buffer = bytearray()
        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
            if not self.running:
                break

            if chunk:
                buffer.extend(chunk)
                if len(buffer) >= self.buffer_size:
                    try:
                        self.audio_queue.put(bytes(buffer), timeout=1)
                    except Exception:
                        logging.warning("Queue full, skipping chunk")
                    buffer.clear()

    def _cleanup(self, session, worker_thread):
        """Clean up resources when stopping the service."""
        self.running = False
        session.close()
        if worker_thread:
            worker_thread.join(timeout=5)


def setup_logging():
    """Configure logging settings and create necessary directories."""
    # Create directory for log and output files if it doesn't exist
    log_dir = os.path.dirname(AudioStreamTranscriber.LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(AudioStreamTranscriber.LOG_PATH),
            logging.StreamHandler(sys.stdout)
        ]
    )


def signal_handler(signum, frame):
    """Handle termination signals."""
    logging.info("Received signal to terminate. Shutting down...")
    sys.exit(0)


def main():
    """Main entry point for the application."""
    signal.signal(signal.SIGTSTP, signal_handler)  # Ctrl+Z
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    
    setup_logging()

    try:
        transcriber = AudioStreamTranscriber()
        transcriber.stream_and_transcribe()
    except ConfigurationError as e:
        logging.error(f"Configuration error: {e}")
        logging.error("Please check your environment variables and try again.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
