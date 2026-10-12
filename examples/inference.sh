#!/usr/bin/env bash
# Run from the repository root after installing MotionGlot and supplying weights.
set -euo pipefail

motionglot generate --task text-to-human --model checkpoints/motionglot \
  --human-vq checkpoints/human_vq/latest.pth --normalization checkpoints/human_vq/normalization.npz \
  --text 'A person walks forward.' --output outputs/text-to-human

motionglot generate --task question-to-human --model checkpoints/motionglot \
  --human-vq checkpoints/human_vq/latest.pth --normalization checkpoints/human_vq/normalization.npz \
  --text 'How does a person practice in a gymnastics class?' --output outputs/question-to-human

motionglot generate --task text-to-robot --model checkpoints/motionglot \
  --robot-vq checkpoints/robot_vq/latest.pth \
  --text 'A robot walks forward and turns right.' --output outputs/text-to-robot

motionglot generate --task goal-to-robot --model checkpoints/motionglot \
  --robot-vq checkpoints/robot_vq/latest.pth --goal 1 2 --output outputs/goal-to-robot

motionglot generate --task human-to-text --model checkpoints/motionglot \
  --human-vq checkpoints/human_vq/latest.pth --normalization checkpoints/human_vq/normalization.npz \
  --motion data/HumanML3D/new_joint_vecs/000000.npy --output outputs/human-to-text

# swap-xy applies to the recovered robot input files before their VQ encoder.
motionglot generate --task robot-to-text --model checkpoints/motionglot \
  --robot-vq checkpoints/robot_vq/latest.pth --motion data/QUAD-LOCO/robot_trajectories/W_8_9.pt \
  --swap-xy --output outputs/robot-to-text
