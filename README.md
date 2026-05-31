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

## Result Export

ระบบไม่ได้แสดงผลเฉพาะบนวิดีโอเท่านั้น แต่ export ผลลัพธ์ออกมาเป็นไฟล์เพื่อให้ตรวจสอบย้อนหลัง วิเคราะห์ต่อ และใช้ประกอบการอธิบายในรอบสัมภาษณ์ได้

Output files:

```txt
outputs/output_video.mp4
outputs/summary.json
outputs/tracks.csv
outputs/events.csv
outputs/performance_report.json
```

### 1. Annotated Video

```txt
outputs/output_video.mp4
```

วิดีโอผลลัพธ์ที่วาดข้อมูลสำคัญลงบนเฟรม เช่น:

- bounding box รอบคน
- `G` = Global ID
- `T` = Tracker ID
- category เช่น `superai_shirt`, `non_superai`, `unknown`
- trajectory
- outside / door / inside zone
- realtime unique count
- enter / exit count
- จำนวน SuperAI / Non-SuperAI / Unknown

ไฟล์นี้ใช้สำหรับดูผลลัพธ์แบบ visual และใช้เป็นวิดีโอประกอบการส่งงาน

### 2. Summary JSON

```txt
outputs/summary.json
```

ไฟล์สรุปผลรวมของระบบ เช่น:

```json
{
  "video": "entrance.mov",
  "method": "YOLO + BoT-SORT + Appearance ReID + Door-Zone Counting + HSV Attributes",
  "total_unique_people": 72,
  "enter_count": 18,
  "exit_count": 51,
  "superai_people": 60,
  "non_superai_people": 11,
  "unknown_people": 1,
  "average_fps": 4.58,
  "total_frames": 2556,
  "processed_frames": 2548
}
```

ไฟล์นี้ใช้สำหรับตอบคำถามหลักของโจทย์ว่า “นับได้กี่คน” และสรุป performance เบื้องต้น

### 3. Tracks CSV

```txt
outputs/tracks.csv
```

ไฟล์นี้เก็บข้อมูลรายคนในระดับ `Global ID`

ตัวอย่าง columns:

```txt
global_id,
track_ids,
category,
first_seen,
last_seen,
first_frame,
last_frame,
counted,
direction,
avg_confidence,
category_confidence
```

ความสำคัญของไฟล์นี้คือช่วยตรวจว่า ReID รวม track หลายตัวเป็นคนเดียวกันได้หรือไม่ เช่น:

```txt
global_id = 12
track_ids = 3|8|15
```

แปลว่า tracker เคยให้ ID 3, 8 และ 15 แต่ระบบมองว่าเป็นคนเดียวกันในระดับ Global ID

### 4. Events CSV

```txt
outputs/events.csv
```

ไฟล์นี้เก็บเหตุการณ์เข้า/ออกที่ระบบตรวจพบ

ตัวอย่าง columns:

```txt
event_id,
global_id,
event_type,
category,
timestamp,
frame_id,
line_crossed,
confidence,
category_confidence
```

ตัวอย่าง event:

```txt
event_id = 5
global_id = 12
event_type = enter
timestamp = 8.42
frame_id = 252
category = superai_shirt
```

ไฟล์นี้ช่วยตรวจสอบว่าแต่ละ enter/exit เกิดขึ้นตอนไหน และเกิดจาก Global ID คนใด

### 5. Performance Report

```txt
outputs/performance_report.json
```

ไฟล์นี้เก็บข้อมูลความเร็วและ latency ของระบบ เช่น:

```json
{
  "processed_frames": 2548,
  "total_processing_time_seconds": 556.67,
  "avg_fps": 4.58,
  "avg_total_time_ms": 200.35,
  "p95_latency_ms": 326.38,
  "avg_tracking_time_ms": 78.05,
  "avg_reid_attribute_time_ms": 82.62,
  "avg_counting_time_ms": 0.13,
  "avg_visualization_time_ms": 22.15
}
```

เหตุผลที่ต้องมี performance report คือไม่ควรบอกแค่ว่า “ระบบเร็ว” แต่ต้องวัดจริงว่าแต่ละส่วนใช้เวลาเท่าไร โดยเฉพาะ:

- tracking
- ReID + attribute classification
- counting
- visualization
- total latency

### Why Export Matters

การ export หลายรูปแบบช่วยให้ระบบตรวจสอบได้มากกว่าการดูวิดีโออย่างเดียว:

- `output_video.mp4` ใช้ดูผลแบบ visual
- `summary.json` ใช้สรุปคำตอบสุดท้าย
- `tracks.csv` ใช้ตรวจ Global ID / ReID
- `events.csv` ใช้ตรวจ enter/exit event
- `performance_report.json` ใช้แสดงความเร็วและความเป็นระบบ

## Review Artifacts

นอกจากวิดีโอ annotated และไฟล์สรุปผล ระบบยัง export ภาพ crop สำหรับ manual review เพื่อช่วยตรวจสอบคุณภาพของ Global ID, ReID และ Attribute Classification

เหตุผลที่ต้องมีส่วนนี้ เพราะระบบนับคนจากวิดีโอจริงมีความไม่แน่นอน เช่น คนถูกบัง, tracker เปลี่ยน ID, เสื้อเห็นไม่ชัด หรือคนไม่ได้ใส่เสื้อ SuperAI แต่มีป้าย/สายคล้องคอ การมี review artifacts ช่วยให้ตรวจผลย้อนหลังได้เป็นระบบมากกว่าการดูวิดีโออย่างเดียว

### 1. Head Crops for Identity Review

```txt
outputs/head_crops/
outputs/head_contact_sheet.jpg
```

ระบบ crop บริเวณศีรษะโดยประมาณจาก person bounding box แล้วบันทึกแยกตาม Global ID เช่น:

```txt
outputs/head_crops/G001/
outputs/head_crops/G002/
outputs/head_crops/G003/
```

ตัวอย่างไฟล์:

```txt
outputs/head_crops/G012/frame_000420_T31.jpg
```

ความหมาย:

- `G012` = Global ID ที่ระบบใช้เป็น anonymous person ID
- `frame_000420` = frame ที่ crop มาจากวิดีโอ
- `T31` = tracker ID ตอนนั้น

`head_contact_sheet.jpg` คือภาพรวมที่ดึงตัวอย่าง head crop ของแต่ละ Global ID มาเรียงกัน เพื่อให้ตรวจเร็วว่า Global ID ซ้ำหรือแตกผิดหรือไม่

จุดประสงค์ของ Head Crops:

- ตรวจว่า Global ID เดียวกันยังดูเป็นคนเดิมหรือไม่
- ตรวจว่าคนเดียวกันถูกแยกเป็นหลาย Global ID หรือเปล่า
- ตรวจว่า ReID merge คนผิดหรือไม่
- ใช้ประกอบการอธิบายว่าระบบมี manual audit process

หมายเหตุสำคัญ:

ระบบนี้ไม่ได้ทำ automatic face recognition และไม่ได้ใช้ใบหน้าเป็น biometric identity matching ภาพ head crop เป็นเพียง artifact สำหรับให้มนุษย์ตรวจสอบย้อนหลังด้วยสายตาเท่านั้น

### 2. Review Crops for Attribute Classification

```txt
outputs/review_crops/
outputs/review_crops/non_superai_contact_sheet.jpg
outputs/review_crops/unknown_contact_sheet.jpg
```

ระบบ export full-body crop สำหรับ category ที่ควรตรวจเพิ่ม เช่น:

- `non_superai`
- `unknown`

ตัวอย่างโครงสร้างไฟล์:

```txt
outputs/review_crops/non_superai/G004/frame_000104_T10.jpg
outputs/review_crops/unknown/G021/frame_000660_T53.jpg
```

เหตุผลที่ต้อง export กลุ่มนี้:

- `non_superai` ควรแปลว่า “มั่นใจว่าไม่ใช่ SuperAI” จึงควรตรวจได้
- `unknown` คือกลุ่มที่ระบบไม่มั่นใจ ไม่ควรเดาสุ่ม
- บางคนอาจไม่ได้ใส่เสื้อ SuperAI แต่มีป้ายหรือสายคล้องคอ
- บางเฟรมอาจ crop เสื้อไม่ครบ หรือถูกบัง
- แสงและ motion blur อาจทำให้ HSV classifier สับสน

ใน review crop จะมี label เช่น:

```txt
G004 T10 non_superai det=0.36 attr=0.88
```

ความหมาย:

- `G004` = Global ID
- `T10` = Tracker ID
- `non_superai` = category ที่ระบบให้
- `det` = detection confidence
- `attr` = attribute confidence

### 3. Contact Sheets

ระบบสร้าง contact sheet เพื่อดูหลายคนพร้อมกันได้รวดเร็ว

```txt
outputs/head_contact_sheet.jpg
outputs/review_crops/non_superai_contact_sheet.jpg
outputs/review_crops/unknown_contact_sheet.jpg
```

ประโยชน์:

- ตรวจคนจำนวนมากได้เร็ว
- เห็น pattern ความผิดพลาดของระบบ
- ใช้ประกอบการ tuning threshold
- ใช้เป็นหลักฐานว่า pipeline ไม่ได้แค่ print count แต่มีขั้นตอนตรวจสอบผล

### 4. Why This Matters

Review artifacts ทำให้ระบบมีความน่าเชื่อถือขึ้น เพราะสามารถตอบคำถามเหล่านี้ได้:

```txt
คนนี้ถูกนับซ้ำไหม?
Global ID นี้เป็นคนเดียวกันจริงไหม?
คนที่เป็น non_superai ถูกจัดถูกหรือเปล่า?
unknown เกิดจากอะไร?
event เข้า/ออกมาจากคนคนไหน?
```

## Code Architecture

โค้ดถูกจัดเป็น layer เพื่อให้แยกหน้าที่ชัดเจน อ่านง่าย และต่อยอดได้เหมือนโปรเจกต์จริง ไม่ได้รวมทุกอย่างไว้ใน `main.py`

ภาพรวม flow:

```txt
main.py
  -> load config
  -> setup logger
  -> start PeopleAnalyticsApp

PeopleAnalyticsApp
  -> read video frame
  -> run tracker
  -> run attribute classifier
  -> assign Global ID
  -> update counter
  -> render annotated frame
  -> write video
  -> export JSON / CSV / performance report
```

### Folder Responsibility

```txt
src/
├── app/
│   ├── runner.py
│   ├── factories.py
│   └── summaries.py
├── config/
│   └── settings.py
├── domain/
│   └── __init__.py
├── io/
│   ├── artifacts.py
│   └── video.py
├── observability/
│   └── logging.py
├── vision/
│   ├── attributes.py
│   ├── identity.py
│   ├── reid.py
│   └── tracking.py
├── visualization/
│   └── renderer.py
├── counter.py
├── global_id_manager.py
├── tracker.py
├── reid.py
├── attribute_classifier.py
├── exporter.py
├── metrics.py
├── preview.py
├── video_reader.py
└── utils.py
```

### `main.py`

`main.py` เป็น entrypoint เท่านั้น มีหน้าที่:

- parse argument เช่น `--config`
- setup logging
- load config
- start application runner

เหตุผลคือไม่ควรให้ `main.py` มี logic ของ Computer Vision เยอะเกินไป เพราะจะ maintain ยาก

### `src/app/runner.py`

เป็น orchestration หลักของระบบ

หน้าที่:

- เปิดวิดีโอ
- loop ทีละ frame
- เรียก tracker
- เรียก attribute classifier
- เรียก Global ID manager
- update counter
- render video
- export result ตอนจบ

ไฟล์นี้ควบคุมลำดับการทำงานของ pipeline แต่ไม่ได้เก็บรายละเอียด algorithm ลึก ๆ ไว้เอง

### `src/app/factories.py`

ใช้สร้าง component จาก `config.yaml`

เช่น:

- tracker
- ReID model
- Global ID manager
- counter
- renderer
- exporter
- review crop exporter

ข้อดีคือถ้าจะเปลี่ยน tracker หรือ parameter ไม่ต้องไปแก้หลายที่

### `src/app/summaries.py`

รวม logic สำหรับสร้าง summary ระหว่างรันและหลังรัน

เช่น:

- filter Global ID ที่ valid
- filter event เฉพาะคนที่ผ่านเงื่อนไข
- สร้าง live count
- สร้าง final count
- นับ category summary

แยกไฟล์นี้ออกมาเพราะ summary logic เป็น business logic ไม่ควรปนกับ drawing หรือ tracking

### `src/config/settings.py`

โหลดและ validate config

หน้าที่:

- อ่าน `config.yaml`
- ตรวจว่ามี section สำคัญครบ
- ส่ง config ให้ application ใช้งาน

ตัวอย่าง section ที่ต้องมี:

```txt
video
detection
tracking
reid
counting
attribute
export
```

### `src/vision/`

เป็น public interface ของฝั่ง Computer Vision

```txt
src/vision/tracking.py
src/vision/reid.py
src/vision/identity.py
src/vision/attributes.py
```

หน้าที่:

- `tracking.py`: YOLO + BoT-SORT tracking
- `reid.py`: appearance feature extraction
- `identity.py`: Global ID manager
- `attributes.py`: SuperAI / Non-SuperAI / Unknown classifier

โฟลเดอร์นี้ทำให้ reviewer เห็นชัดว่าส่วน Computer Vision อยู่ตรงไหน

### `src/counter.py`

เก็บ logic การนับทั้งหมด

รองรับ:

- line crossing
- zone sequence
- outside / door / inside
- lost-at-door event
- enter / exit event
- unique people state

เหตุผลที่แยกออกมา เพราะ counting logic เป็นหัวใจของโจทย์ และควรอ่านแยกจาก YOLO/visualization ได้

### `src/visualization/`

ใช้วาด annotated video

แสดง:

- bounding box
- Global ID
- Track ID
- category
- trajectory
- zone polygon
- count dashboard
- SuperAI / Non-SuperAI / Unknown summary

ส่วนนี้แยกออกจาก logic นับคน เพื่อไม่ให้ visualization ไปปนกับ decision logic

### `src/io/`

รวม input/output adapters

หน้าที่:

- video reader
- video writer
- result exporter
- head crop exporter
- review crop exporter

แยกไว้เพราะ I/O เป็นเรื่องของไฟล์ ไม่ใช่ algorithm

### `src/observability/`

ดูแล logging และ runtime visibility

ระบบ log ไปที่:

```txt
outputs/logs/run.log
```

Log ใช้ดู:

- video metadata
- model config
- final count
- performance
- output paths

### `src/domain/`

รวม domain object ที่ใช้ข้าม module เช่น:

- `CountSummary`
- `CountEvent`
- `GlobalObservation`
- `GlobalPersonProfile`

ช่วยให้ module อื่น import object สำคัญได้จากจุดเดียว

## Why This Architecture

เหตุผลที่แยกแบบนี้:

- `main.py` สั้นและเข้าใจง่าย
- Computer Vision logic แยกจาก application flow
- Counting logic อ่านแยกได้
- Visualization ไม่ปนกับ decision logic
- Export และ logging แยกเป็น infrastructure
- สามารถเปลี่ยน tracker, ReID, classifier หรือ counter ได้ง่าย
- เหมาะกับการอธิบายใน interview เพราะแสดง system design ชัดเจน

## Runtime Sequence

```txt
1. main.py loads config
2. PeopleAnalyticsApp starts
3. VideoReader reads frame
4. MultiObjectTracker detects and tracks people
5. HSVAttributeClassifier predicts frame-level category
6. GlobalIDManager assigns Global ID
7. ZoneSequenceCounter updates enter/exit state
8. TrackingPreview renders annotated frame
9. DebugVideoWriter writes video
10. ResultExporter saves JSON/CSV
11. PerformanceMeter saves runtime report
```

## Design Principle

ระบบนี้ตั้งใจแยก `what to decide` ออกจาก `how to display`

ตัวอย่าง:

- การตัดสินว่าเป็นคนเดิมหรือไม่ อยู่ใน `GlobalIDManager`
- การตัดสินว่าเข้า/ออก อยู่ใน `counter.py`
- การตัดสินว่าเป็น SuperAI หรือไม่ อยู่ใน `attribute_classifier.py`
- การวาดผล อยู่ใน `preview.py`
- การ export ผล อยู่ใน `exporter.py`
