# 🧠 PerceptionLab

![Status](https://img.shields.io/badge/status-WIP-orange?style=flat-square)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/license-GPLv3-blue?style=flat-square)

> **Build your own SLAM system — swap components, target any device.**
> A modular, plug-and-play perception pipeline for robotics and computer vision research.

---

## 📌 Overview

**PerceptionLab** is a modular perception framework designed for robotics and computer vision research. It lets you assemble your own SLAM (Simultaneous Localization and Mapping) pipeline by plugging in the components you want — feature extractors, optimizers, trackers, depth estimators, and more.

> ⚠️ **Work In Progress** — The framework is under active development. Some modules may not be fully functional yet.

---

## ✨ Features

| Module | Description |
|---|---|
| 🔍 Feature Extraction | Supports ORB, SIFT, SuperPoint, and custom extractors |
| 📐 Nonlinear Optimization | Integrated with [g2o](https://github.com/RainerKuemmerle/g2o); designed to support Ceres & GTSAM |
| 🎯 Visual Tracking | Descriptor-based tracker with configurable reprojection error & confidence |
| 📏 Depth Estimation | Stereo depth estimator with configurable camera IDs |
| 🗺️ Map Management | Atlas-style map with keyframe decisions and local bundle adjustment |
| ⚙️ Config-Driven | Fully YAML-configurable sensors, pipeline, and algorithm parameters |
| 🔗 Python + C++ Hybrid | C++ core with Python bindings via pybind11 |

---

## 🏗️ Architecture
```
PerceptionLab/
├── src/
│   ├── atlas/             # Map management & keyframe storage
│   ├── core/              # Core SLAM logic
│   ├── config/            # Config loading & validation
│   ├── depth_estimator/   # Stereo / depth estimation modules
│   ├── feature_extractor/ # ORB, SIFT, SuperPoint, etc.
│   ├── interfaces/        # Sensor interfaces (CSV reader, etc.)
│   ├── motion_model/      # Motion prediction models
│   ├── optimizer/         # g2o-based graph optimization
│   ├── pipeline/          # Main pipeline orchestration
│   ├── sensors/           # Camera sensor abstraction
│   ├── tracking/          # Feature tracking & pose estimation
│   ├── utils/             # Shared utilities
│   └── visualize.py       # Trajectory & map visualization
├── test/                  # Unit & integration tests
├── config.yaml            # Pipeline configuration
├── CMakeLists.txt         # C++ build configuration
├── build.sh               # Build script
└── requirements.txt       # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/srikanth-444/PerceptionLab.git
cd PerceptionLab
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure the C++ build

Open `CMakeLists.txt` and update the path to your local `g2o` installation:
```cmake
# Example — update this to match your system
set(g2o_DIR "/path/to/your/g2o/lib/cmake/g2o")
```

> **Note:** You may need to [build g2o from source](https://github.com/RainerKuemmerle/g2o).

### 4. Build
```bash
bash build.sh
```

---

## ⚙️ Configuration

All pipeline parameters are controlled through `config.yaml`. Key sections:
```yaml
sensors:
  cameras:
    - ID: 0
      intrinsic: { fx: 458.654, fy: 457.296, cx: 367.215, cy: 248.375 }
      camera_model: pinhole
      rate_hz: 20
      resolution: [752, 480]
      interface:
        type: "csv_reader"
        path: "/path/to/your/cam0/data.csv"

pipe_line:
  visual_odometry:
    feature_extractor:
      type: "ORB"
      nfeatures: 1200
    depth_estimator:
      type: "stereo"
    tracker:
      type: "descriptor"
      reprojection_error: 5.0
```

---

## 📦 Dependencies

### Python

| Package | Version |
|---|---|
| numpy | >= 1.26.0 |
| opencv-python | > 4.8.0 |
| open3d | >= 0.17.1 |
| scipy | >= 1.12.0 |
| PyYAML | >= 6.0.0 |
| scikit-learn | >= 1.3.0 |

### System / C++

- C++17 compatible compiler (GCC 10+ or Clang 12+)
- OpenCV >= 4.8
- Eigen >= 3.4
- [g2o](https://github.com/RainerKuemmerle/g2o) (graph optimization)
- pybind11 (Python bindings)

---

## 🛣️ Roadmap

- [x] Base sequential pipeline
- [x] ORB feature extractor
- [x] g2o optimizer integration
- [x] Stereo depth estimator
- [x] Local bundle adjustment
- [x] Tracking < 20ms
- [ ] Parallel pipeline (frame processing, tracking, map updates)
- [ ] SuperPoint / learned feature extractor support
- [ ] Ceres / GTSAM optimizer backends
- [ ] ROS2 interface
- [ ] Loop closure detection

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests to suggest improvements, report bugs, or add new components.

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

This project is inspired by and partially based on:
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) — Campos et al., GPLv3 — for the Atlas map management design, tracking pipeline, and local bundle adjustment concepts.
- [g2o](https://github.com/RainerKuemmerle/g2o) — BSD License — used for graph-based nonlinear optimization.

---

## 👤 Author

**srikanth-444** — [@srikanth-444](https://github.com/srikanth-444)

⭐ Star this repo if you find it useful!
```

---

## 📄 `LICENSE` (new file — GPLv3)
```
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2026 srikanth-444
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  When we speak of free software, we are referring to freedom, not
price. Our General Public Licenses are designed to make sure that you
have the freedom to share and change all versions of a program--to
make sure it remains free software for all its users.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights. Therefore, you
have certain responsibilities if you distribute copies of the
software, or if you modify it: responsibilities to respect the
freedom of others.

  For the full terms and conditions, see:
  https://www.gnu.org/licenses/gpl-3.0.txt

--------------------------------------------------------------------------------

PerceptionLab — Copyright (C) 2026 srikanth-444

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------

THIRD-PARTY ATTRIBUTIONS

This project is inspired by and partially based upon the following works:

1. ORB-SLAM3
   Authors: Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez,
            José M. M. Montiel, Juan D. Tardós
   Repository: https://github.com/UZ-SLAMLab/ORB_SLAM3
   License: GNU General Public License v3.0
   Used for: Atlas-style map management design, visual tracking pipeline
             architecture, and local bundle adjustment concepts.

2. g2o — General Graph Optimization
   Authors: Rainer Kümmerle et al.
   Repository: https://github.com/RainerKuemmerle/g2o
   License: BSD License
   Used for: Graph-based nonlinear optimization backend.