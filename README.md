# EAF-Net

---

## 项目简介
本仓库为3D 点云语义分割论文核心组件实现，仅包含论文提出的 EAF-Net 与 DWS核心代码，可集成到现有 3D 点云语义分割框架中作为模块使用。
---

## 环境依赖
```bash
pip install -r requirements.txt
```
---

## 目录结构
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

## 注意事项
- 代码基于 **TensorFlow 1.x** 开发，已禁用TF2.x行为
- 需提前编译C++扩展
- 显存不足可调小 `batch_size` / `num_points` 参数