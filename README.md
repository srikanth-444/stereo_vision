# PerceptionLab (WIP)

**PerceptionLab** is a modular perception pipeline designed for robotics and computer vision research.  
**Work in Progress:** the framework is under active development to allow users to build their own perception system by plugging in their preferred components.

---

## Planned Features

- **Modular Feature Extractors**
  - Support for custom extractors like SuperPoint, ORB, SIFT, etc. 
  
- **Nonlinear Optimizers**
  - Integration with **g2o**.
  - Designed to allow adding other optimizers (Ceres, GTSAM) in the future.

- **Parallel Processing Pipeline**
  - Decoupled stages for efficiency:
    - Frame processing
    - Tracking → Keyframe decision → New keyframe
    - Map updates (optional, separate)

- **Customizable Components**
  - Users can plug in different feature extractors, optimizers, keyframe strategies, and map update logic.

---

## Current Status

- Base sequential pipeline implemented.  
- Planning parallelization for frame processing, tracking, and map updates.  
- Feature extractor and optimizer modules under design.  

---
## Requirements

PerceptionLab depends on both Python and C++ libraries. Make sure the following are installed:

### System / C++ Dependencies
- **C++17 compatible compiler** (e.g., GCC 10+, Clang 12+)
- **OpenCV** (>=4.8)
- **Eigen** (>=3.4)
- **g2o** (graph optimization library)
- **pybind11** (for Python bindings)

> You may need to build g2o from source.

## Getting Started

> This is a work in progress. Some modules may not be fully functional.

```bash
git clone https://github.com/srikanth-444/PerceptionLab.git
cd PerceptionLab
pip install -r requirements.txt
```
point the libraries in the main CMakeLists file for g2o library 
```bash
bash build.sh
```
