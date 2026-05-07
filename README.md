# EAF-Net

---

## Project Introduction
This repository contains the core component implementation of the 3D point cloud semantic segmentation paper, including only the core code of EAF-Net and DWS proposed in the paper. It can be integrated into existing 3D point cloud semantic segmentation frameworks as a module.
---

## Environment Dependencies
```bash
pip install -r requirements.txt
```
---

## Directory Structure
```
.
├── DWS.py              
├── EAF-Net.py          
├── helper_tool.py   
├── tester_S3DIS.py     
├── tester_Toronto3D.py 
├── requirements.txt   
└── README.md
```
---

## Notes
- The code is developed based on TensorFlow 1.x, and TF2.x behaviors have been disabled.
- The C++ extension needs to be compiled in advance.
- If the video memory is insufficient, adjust the batch_size / num_points parameters to smaller values.