# Smart Entrance People Analytics

ระบบนับคนบริเวณทางเข้า/ออกอาคารแบบไม่ซ้ำ

## Objective

โจทย์ต้องการนับจำนวนคนในวิดีโอทางเข้า/ออกอาคาร โดยต้องส่ง source code, README, requirements, video result และอธิบายแนวคิดได้ในการสัมภาษณ์

## Why Not Count Bounding Boxes?

ถ้านับ bounding box ต่อเฟรม คนหนึ่งคนที่เดินผ่านกล้องหลายเฟรมจะถูกนับซ้ำจำนวนมาก ระบบนี้จึงออกแบบเป็น pipeline:

```txt
Video
-> Person Detection
-> Multi-Object Tracking
-> ReID / Global ID
-> Door-Zone Counting
-> Attribute Classification
-> Visualization
-> Export Results
```

## Planned Features

- YOLO person detection
- BoT-SORT / ByteTrack tracking
- Global anonymous person ID
- Appearance-based ReID
- Door-zone enter/exit counting
- SuperAI / Non-SuperAI / Unknown classification
- Annotated video output
- JSON / CSV / performance report
- Review crops for manual audit

## Project Structure

```txt
people-counting-reid/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
├── src/
│   ├── app/
│   ├── config/
│   ├── domain/
│   ├── io/
│   ├── observability/
│   ├── vision/
│   └── visualization/
├── outputs/
└── assets/
```

- YAML config loading
- required config validation
- video metadata reader
- optional frame resizing
- process every N frames support

Run:

```bash
python main.py
```

Expected output:

```txt
Smart Entrance People Analytics
Step 2: video reader and config loader completed.

Config: config.yaml
Processing video: entrance.mov
FPS: ...
Original size: ...
Total frames: ...

First processed frame:
Frame ID: 0
Timestamp: 0.00s
Processed size: ...
```