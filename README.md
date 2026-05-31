# Smart Entrance People Analytics

ระบบวิเคราะห์วิดีโอทางเข้า/ออกอาคารสำหรับนับจำนวนคนแบบไม่ซ้ำ โดยใช้ Computer Vision pipeline ที่มากกว่าแค่ detect คนแล้วนับ bounding box

โปรเจกต์นี้ทำสำหรับโจทย์ People Counting จากวิดีโอ โดยเน้นแนวคิดระบบจริง: อ่านวิดีโอ, ตรวจจับคน, tracking, นับเข้า/ออกจาก zone และลดการนับซ้ำด้วย Global ID

## Problem

ถ้านับจำนวนคนจาก bounding box ในแต่ละเฟรมโดยตรง ผลลัพธ์จะผิดง่ายมาก เพราะคนหนึ่งคนอยู่ในวิดีโอหลายเฟรม เช่น คนเดียวเดินผ่านกล้อง 100 เฟรม ถ้านับทุก detection ก็อาจกลายเป็น 100 คน

ดังนั้นระบบนี้ไม่ได้นับจากจำนวน box ต่อเฟรม แต่ใช้ flow แบบนี้:

```txt
Video
-> YOLO Person Tracking
-> Track ID
-> Appearance ReID
-> Global ID
-> Door Zone Counting
-> Annotated Video
```

## Current Features

- อ่านวิดีโอจาก `config.yaml`
- รองรับ resize frame เพื่อเพิ่มความเร็ว
- ตรวจจับและ track คนด้วย YOLO + BoT-SORT
- แสดง bounding box, track ID, global ID และ trajectory
- ใช้ zone บริเวณประตูเพื่อแยก enter / exit
- ใช้ Global ID เพื่อลดการนับซ้ำเมื่อ tracker เปลี่ยน ID
- สร้างวิดีโอผลลัพธ์ annotated video
- แยก logic เป็นไฟล์ย่อยเพื่อให้อ่านและต่อยอดง่าย

## System Overview

```txt
entrance.mov
   |
   v
VideoReader
   |
   v
YOLO + BoT-SORT Tracking
   |
   v
Tracked Objects (T ID)
   |
   v
Appearance ReID
   |
   v
Global ID Manager (G ID)
   |
   v
Door Zone Counter
   |
   v
Annotated Video Output
```

## Project Structure

```txt
people-counting-reid/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── domain/
│   │   └── __init__.py
│   ├── io/
│   │   └── video.py
│   ├── vision/
│   │   ├── tracking.py
│   │   ├── reid.py
│   │   └── identity.py
│   ├── counter.py
│   ├── debug_video_writer.py
│   ├── global_id_manager.py
│   ├── preview.py
│   ├── reid.py
│   ├── tracker.py
│   ├── utils.py
│   └── video_reader.py
├── outputs/
└── assets/
```

## Main Logic

### 1. Video Reader

`VideoReader` เปิดวิดีโอจาก path ใน `config.yaml` แล้วดึงข้อมูลพื้นฐาน เช่น:

- FPS
- width / height
- total frames
- timestamp ของแต่ละ frame

รองรับการ resize ด้วย `resize_width` เพื่อให้รันเร็วขึ้น

### 2. YOLO Person Tracking

ระบบใช้ Ultralytics YOLO ผ่าน `model.track()` เพื่อ detect และ track คนในวิดีโอ

ค่าที่ได้ต่อคน:

- bounding box
- confidence
- track ID
- center point
- foot point
- frame id
- timestamp

ในวิดีโอจะเห็น label ประมาณนี้:

```txt
T12
```

`T` คือ Tracker ID ซึ่งเป็น ID ชั่วคราวจาก tracker

### 3. Why Track ID Is Not Enough

Tracker ID อาจเปลี่ยนได้ เช่น:

- คนถูกบัง
- คนเดินออกจากเฟรมแล้วกลับมา
- คนเดินผ่านประตูแล้ว tracker หลุด
- คนหลายคนเดินใกล้กัน

ถ้าใช้ track ID นับคนโดยตรง อาจนับคนเดิมซ้ำได้

### 4. Global ID / ReID

ระบบจึงสร้าง `Global ID` เพื่อแทนคนหนึ่งคนในระดับระบบ

ในวิดีโอจะเห็น:

```txt
G8 T21
```

ความหมาย:

- `G` = Global ID ที่ระบบใช้สำหรับนับคน
- `T` = Track ID จาก tracker

การจับว่า track ใหม่เป็นคนเดิมหรือไม่ ใช้ lightweight appearance ReID จากภาพคน เช่น:

- สีเสื้อบริเวณ torso
- สีกางเกง
- สีรองเท้า
- บริเวณศีรษะ/ผมแบบหยาบ
- shape ของ bounding box
- ระยะเวลาและตำแหน่งล่าสุดที่เห็น

ระบบนี้ไม่ใช้ face recognition และไม่ใช้ข้อมูล biometric

### 5. Door Zone Counting

ระบบไม่ได้นับทันทีที่เห็นคน แต่จะนับจากการเคลื่อนที่ผ่าน zone

ใน config มี 3 zone หลัก:

```txt
inside <-> door <-> outside
```

ตัวอย่าง logic:

```txt
outside -> door -> inside = enter
inside -> door -> outside = exit
outside -> door -> lost = enter
inside -> door -> lost = exit
```

สาเหตุที่ใช้ zone แทน line เส้นเดียว เพราะในวิดีโอจริงบางคนเดินเข้าประตูแล้วหายจากกล้องก่อนจะข้ามเส้นชัดเจน การใช้ door zone ทำให้เหมาะกับสถานการณ์จริงกว่า

### 6. Visualization

ระบบสร้างวิดีโอ annotated output โดยแสดง:

- bounding box รอบคน
- `G` และ `T`
- detection confidence
- trajectory จากจุดเท้า
- zone polygon
- unique count
- enter / exit count
- currently visible people

Output หลัก:

```txt
outputs/debug_tracking_preview.mp4
```

## Configuration

ค่าหลักอยู่ใน `config.yaml`

ตัวอย่าง:

```yaml
video:
  input_path: "entrance.mov"
  output_path: "outputs/output_video.mp4"
  resize_width: 1280
  process_every_n_frames: 1

detection:
  model_name: "yolov8n.pt"
  confidence_threshold: 0.35
  iou_threshold: 0.5
  min_box_area: 1000

tracking:
  tracker_type: "botsort"
  min_track_length: 5
  min_avg_confidence: 0.35

reid:
  enabled: true
  method: "multi_region_appearance"
  appearance_threshold: 0.752
  max_time_gap_seconds: 35.0
  max_spatial_distance: 650

counting:
  mode: "zone"
  disappear_after_frames: 8
```

## Installation

สร้าง virtual environment:

```bash
python -m venv .venv
```

เปิดใช้งานบน Git Bash:

```bash
source .venv/Scripts/activate
```

ติดตั้ง dependencies:

```bash
python -m pip install -r requirements.txt
```

## How to Run

วางไฟล์วิดีโอไว้ที่ root project:

```txt
entrance.mov
```

รันแบบเร็ว:

```bash
python main.py --max-frames 300
```

รันเต็ม:

```bash
python main.py
```

ถ้าใช้ `.venv` โดยตรง:

```bash
./.venv/Scripts/python.exe main.py
```

## Current Output

หลังรันจะได้วิดีโอ:

```txt
outputs/debug_tracking_preview.mp4
```

Console จะแสดงประมาณนี้:

```txt
Smart Entrance People Analytics
Processing video: entrance.mov
Detector model: yolov8n.pt
Tracker: botsort
ReID enabled: True
Counting mode: zone

Identity-aware counting test completed.
Total Unique People: ...
Enter Count: ...
Exit Count: ...
```

## Important Notes

ตอนนี้ระบบนับจาก `Global ID` แล้ว ไม่ใช่ raw bounding box และไม่ใช่ raw track ID โดยตรง

อย่างไรก็ตาม ReID ตอนนี้เป็นแบบ lightweight color/appearance feature ยังไม่ใช่ deep ReID model เช่น OSNet หรือ FastReID ดังนั้นถ้าคนแต่งตัวคล้ายกันมาก หรือถูกบังหนักมาก อาจยังมี ID switch หรือ merge ผิดได้

## Limitations

- ถ้าคนถูกบังหนักมาก tracker อาจเปลี่ยน ID
- ถ้าคนใส่เสื้อ/กางเกงคล้ายกันมาก ReID อาจสับสน
- zone ต้องปรับให้ตรงกับมุมกล้องจริง
- color-based ReID ยังไม่แม่นเท่า deep ReID model
- ยังไม่ได้ export JSON/CSV ในเวอร์ชันนี้

## Next Improvements

- เพิ่ม attribute classification สำหรับ SuperAI / Non-SuperAI / Unknown
- เพิ่ม export `summary.json`, `tracks.csv`, `events.csv`
- เพิ่ม performance report
- เพิ่ม review crops สำหรับตรวจสอบคนที่ระบบไม่มั่นใจ
- ปรับ visualization ให้เหมือน final dashboard
- รองรับ ReID model จริง เช่น OSNet หรือ FastReID

## Attribute Classification

ระบบแยกประเภทคนจากลักษณะเสื้อ โดยใช้ rule-based HSV classifier เป็น baseline ที่รันง่ายและอธิบายได้ชัดเจน

แนวคิดคือไม่ classify จากทั้งภาพ เพราะทั้งภาพมี background, ประตู, ผนัง และคนอื่นปนอยู่ ระบบจะ crop เฉพาะบริเวณ upper body ของ bounding box ก่อน แล้วค่อยวิเคราะห์สี

### Classes

ระบบแบ่งเป็น 3 กลุ่ม:

- `superai_shirt`: มีหลักฐานสีน้ำเงิน/ฟ้าของเสื้อ SuperAI ชัดเจน
- `non_superai`: มั่นใจว่าไม่ใช่เสื้อ SuperAI
- `unknown`: ยังไม่มั่นใจ เช่น โดนบัง แสงไม่ดี เห็นเสื้อไม่ครบ หรือมีสีน้ำเงินเล็กน้อยแต่ไม่พอพิสูจน์

> `non_superai` ไม่ได้แปลว่า unknown แต่แปลว่า “ระบบมั่นใจว่าไม่ใช่ SuperAI”
>
> ถ้าหลักฐานไม่พอ ระบบจะเลือก `unknown` แทนการเดาสุ่ม

### How It Works

ขั้นตอนต่อเฟรม:

```txt
person bbox
-> crop upper body
-> convert BGR to HSV
-> calculate blue pixel ratio
-> calculate dark/non-blue evidence
-> return frame-level category
```

จากนั้นระบบไม่ใช้ผลจากเฟรมเดียวทันที แต่เก็บ vote ไว้ใน `GlobalPersonProfile`

```txt
frame-level prediction
-> category vote per Global ID
-> majority / confidence rule
-> final category per person
```

ตัวอย่าง:

```txt
G12:
frame 101 = superai_shirt
frame 102 = superai_shirt
frame 103 = unknown
frame 104 = superai_shirt

final category = superai_shirt
```

อีกตัวอย่าง:

```txt
G25:
frame 320 = non_superai
frame 321 = non_superai
frame 322 = unknown
frame 323 = non_superai

final category = non_superai
```

### Why Majority Vote

การใช้เฟรมเดียวอาจผิดได้จากหลายสาเหตุ:

- คนถูกบังบางส่วน
- motion blur
- แสงเปลี่ยน
- crop ไปโดนแขนหรือป้าย
- เสื้อ SuperAI เห็นแค่บางส่วน
- background มีสีใกล้เคียงเสื้อ

ดังนั้นระบบจึงใช้หลายเฟรมเพื่อให้ category เสถียรกว่า

### Decision Logic

หลักการตัดสินแบบย่อ:

```txt
ถ้า blue ratio สูงพอ -> superai_shirt
ถ้ามีหลักฐาน non-blue ชัด และ blue ต่ำ -> non_superai
ถ้าหลักฐานไม่พอหรือคลุมเครือ -> unknown
```

ในระดับ Global ID:

```txt
ถ้ามี SuperAI votes มากพอ -> superai_shirt
ถ้ามี non-SuperAI votes ต่อเนื่องพอ และ SuperAI evidence ไม่เด่น -> non_superai
ถ้ายังไม่ชัด -> unknown
```

### Dashboard

วิดีโอ output จะแสดงจำนวนรวมของ category ที่ผ่าน Global ID filtering แล้ว:

```txt
SuperAI
Non-SuperAI
Unknown
```

จำนวนนี้นับจาก `Global ID` ไม่ใช่จำนวน bounding box ในเฟรม

### Limitation

วิธีนี้เป็น color-based classifier จึงยังมีข้อจำกัด:

- ถ้าแสงเพี้ยน สีเสื้ออาจผิด
- ถ้าเสื้อถูกบังมาก ระบบอาจให้ `unknown`
- ถ้าคนไม่ได้ใส่เสื้อ SuperAI แต่มีป้ายหรือสายคล้อง ระบบอาจต้องใช้ review crop หรือ classifier ที่ train เพิ่ม
- ถ้าเสื้อสีคล้าย SuperAI อาจเกิด false positive ได้