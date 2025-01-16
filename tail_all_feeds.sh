#!/bin/bash

# Start a new tmux session (detached)
tmux new-session -d -s transcribe

# Enable pane borders with titles at the top
tmux set-option -g pane-border-status top

# Set fancy formatting for the pane titles
# #[fg=colour39,bold] sets bright blue color and bold
# #[default] resets the formatting
tmux set-option -g pane-border-format "#[fg=colour39,bold]#T#[default]"

# For each feed, create a pane and run tail
first=true
jq -c '.feeds[]' feeds.json | while read -r feed; do
    id=$(echo $feed | jq -r '.id')
    name=$(echo $feed | jq -r '.name')
    
    if [ "$first" = true ]; then
        first=false
    else
        tmux split-window -v
        tmux select-layout tiled
    fi
    
    # Set the pane title
    tmux select-pane -T "${id}_${name}"
    
    # Send tail command to current pane
    tmux send-keys "tail -f feeds/${id}_${name}/transcribe.txt" C-m
done

# Attach to the session
tmux attach-session -t transcribe
