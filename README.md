# 🧠 Multi-Task Vision Toolbox (MTVT)

**A pluggable, augmentation-aware, multi-task computer vision framework**  
built on top of [OpenMMLab](https://openmmlab.com/) — supporting object detection, segmentation, keypoints, depth, and lane line detection in one unified pipeline.

---

## 🚀 Why MTVT?
Most vision toolkits are **single-task** or fragmented across separate repos.  
MTVT is designed to let you **mix and match**:

- **Any backbone** from OpenMMLab (ResNet, Swin, ConvNeXt, CSPDarkNet …)
- **Any neck** (FPN, PAN, etc.)
- **Multiple task heads** (detection, segmentation, keypoints, lane lines, etc)
- **Task-specific augmentations** tied to **strongly-typed data containers**

All in a single training loop.

---

## 📦 Features
- ✅ **Plug & Play Modules** — add/remove heads without changing core code  
- ✅ **Multi-task dataset wrapper** — train on multiple labels per image  
- ✅ **Integrated with MMDetection ecosystem**  
- ✅ **Easy extension** — add new tasks with minimal boilerplate  
## 🚧 Future: 
 - Augmentation-aware Data Structures** — `BBoxes`, `Keypoints`, `LaneLines`, `DepthMaps etc.`  
- full mmkeypoints, mmdepth  and others integration  
- support multi-stage networks

---

## 🏗 Architecture

    ┌───────────────────────────────────────────┐
    │                 Dataset                   │
    │  (e.g., COCO, OpenLane, KITTI)            │
    └───────────────────────────────────────────┘
                       │
                       ▼
    ┌───────────────────────────────────────────┐
    │     MultiTaskDataset (Wrapper)            │
    └───────────────────────────────────────────┘
                       │
                       ▼
    ┌───────────────────────────────────────────┐
    │   Data Structures:                        │
    │   ├── BBoxes                              │
    │   ├── SegmentationMasks                   │
    │   ├── Keypoints                           │
    │   ├── LaneLines                           │
    │   ├── DepthMaps                           │
    │   └── CustomData                          │
    └───────────────────────────────────────────┘
                       │
                       ▼
    ┌───────────────────────────────────────────┐
    │ Augmentations (Task-Aware)                │
    │   e.g., Flip, Rotate, Perspective Warp    │
    │   applied only to relevant task keys      │
    └───────────────────────────────────────────┘
                       │
                       ▼
    ┌───────────────────────────────────────────┐
    │ Backbone + Neck + Multiple Heads          │
    │  (Detection, Segmentation, Lane Lines…)   │
    └───────────────────────────────────────────┘
                       │
                       ▼
    ┌───────────────────────────────────────────┐
    │ Multi-task Loss & Metrics                 │
    └───────────────────────────────────────────┘


## 📌 Roadmap

- [x] Multi-task Model
- [x] Bbox, Keypoints, Lane Keypoints, Instance/Semantic Segmentation, Multi-label classification
- [ ] Augmentation aware datatypes
- [ ] Integrate Remaining Tasks
- [ ] Suport multi-stage models
- [ ]


## 📜 License

This project is released under the [Apache 2.0 license](LICENSE).

---

## 📢 Citation

If you use this toolbox in your work, please consider citing it:

```bibtex
@misc{multi-task-vision-toolbox,
  author = {Muhammad Junaid Ahmad},
  title = {Multi-Task Vision Toolbox},
  year = {2025},
  url = {https://github.com/<your-username>/multi-task-vision-toolbox}
}
```