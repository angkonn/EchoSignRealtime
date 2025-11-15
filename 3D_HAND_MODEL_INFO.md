# 3D Hand Model Visualization

## Overview
The web UI now features a real-time 3D hand model that mirrors your glove movements using Three.js and GSAP animations.

## Implementation Details

### Technology Stack
- **Three.js r157**: 3D rendering engine
- **GSAP 3.12**: Smooth animation library  
- **GLTF Model**: Rigged hand model with proper bone structure

### Hand Model Source
The `hand.glb` file is based on [Kirilbt's hand-armature](https://github.com/Kirilbt/hand-armature) project:
- Created in Blender with proper armature/skeleton
- 22 bones total: 1 wrist + (3-4 bones × 5 fingers)
- Pre-rigged for realistic finger animation

### How It Works

1. **Model Loading**: The GLTF hand model is loaded on page init
2. **Bone Mapping**: Finger bones are mapped to sensor inputs:
   - Thumb: bones[3,4,5]
   - Index: bones[7,8,9]
   - Middle: bones[11,12,13]
   - Ring: bones[15,16,17]
   - Pinky: bones[19,20,21]

3. **Real-time Animation**:
   - Flex sensors (0-1) → Finger bend angles (0-72°)
   - IMU gyroscope → Hand orientation (pitch/yaw/roll)
   - GSAP timelines provide smooth 0.5s transitions

### Controls
- **Mouse**: Click and drag to rotate view
- **Scroll**: Zoom in/out
- **Reset Button**: Return to default camera position

## Fallback Behavior
If `hand.glb` fails to load, the system falls back to a simple geometric hand made of cylinders and boxes.

## Credits
- Hand model architecture: [Gill003/Smart-Sign-Language-Translator-Glove](https://github.com/Gill003/Smart-Sign-Language-Translator-Glove)
- Original hand armature: [Kirilbt/hand-armature](https://github.com/Kirilbt/hand-armature)
- Blender tutorial: [sens_3d on YouTube](https://www.youtube.com/watch?v=ZwsDZNP5m2k&t=2s)
