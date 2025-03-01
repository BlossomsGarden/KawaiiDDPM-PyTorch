#!/bin/bash

SCRIPT_NAME="train.py"
LOG_FILE="nohup.out"

PID=$(ps aux | grep "[p]ython $SCRIPT_NAME" | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "Process $SCRIPT_NAME running (PID: $PID). Restarting..."
    kill -9 $PID
    sleep 2
    nohup python $SCRIPT_NAME >> $LOG_FILE 2>&1 &
    echo "Training Restarted"
else
    nohup python $SCRIPT_NAME >> $LOG_FILE 2>&1 &
    echo "Training Started..."
fi