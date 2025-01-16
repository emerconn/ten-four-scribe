# Ten-Four Scribe

Transcribe Broadcastify feeds using an Nvidia CUDA-capable GPU and OpenAI Whisper.

## How to use

- Copy & edit `feeds.json` from `feeds.json.example`
- Run `setup_feeds.sh` to create `compose.yaml` & feed environment files
- Run `docker compose up -d`
- Stream all transcriptions with `tail_all_feeds.sh`
