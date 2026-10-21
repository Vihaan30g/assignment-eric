#!/usr/bin/env bash
set -e
cd /home/marsuser/assignment-eric
source /home/marsuser/ml_env/bin/activate
source /opt/ros/humble/setup.bash
python -m src.run_realtime "$@"
