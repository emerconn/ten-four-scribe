#!/bin/bash

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed"
    exit 1
fi

# Get global credentials
username=$(jq -r '.credentials.username' feeds.json)
password=$(jq -r '.credentials.password' feeds.json)

# Create feeds directory if it doesn't exist
mkdir -p feeds

# Start compose.yaml with services header
echo "services:" > compose.yaml

# Process each feed
jq -c '.feeds[]' feeds.json | while read -r feed; do
    id=$(echo $feed | jq -r '.id')
    name=$(echo $feed | jq -r '.name')
    model_size=$(echo $feed | jq -r '.model_size')
    feed_dir="feeds/${id}_${name}"
    
    # Create feed directory
    mkdir -p "$feed_dir"
    
    # Create .env file
    cat > "$feed_dir/.env" << EOF
BROADCASTIFY_FEED_ID=${id}
BROADCASTIFY_USERNAME=${username}
BROADCASTIFY_PASSWORD=${password}
WHISPER_MODEL_SIZE=${model_size}
EOF

    # Add to compose.yaml
    cat >> compose.yaml << EOF
  ${id}_${name}:
    image: ghcr.io/emerconn/ten-four-scribe:main
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./feeds/${id}_${name}/:/work/data
    env_file:
      - ./feeds/${id}_${name}/.env

EOF
done
