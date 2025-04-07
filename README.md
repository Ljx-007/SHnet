# SHNet

## Overview
This project implements a hybrid CNN-Transformer model combining MobileNetV3 and ViT architectures for image processing tasks.

## Features
- Hybrid architecture with MBConv blocks (MobileNetV3) and Transformer attention
- Efficient implementation using PyTorch
- Support for both CNN and Transformer operations
### Backbone:<br>
![image](https://github.com/user-attachments/assets/4de3414d-766d-45ea-9b8f-759f375fbb0d)<br><br>
### GateNet
![image](https://github.com/user-attachments/assets/1c4e3259-e3c2-4117-b6de-94d863acdf6f)
## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.SHnet import SHNet
model = SHNet()
# Your training/inference code here
```



## File Structure
```
src/
  SHnet.py - Main model implementation
  module.py - Core modules (MBConv, Attention, etc.)
  efficientnet.py - EfficientNet components
  filters.py - Filter utilities
  utils.py - Helper functions
```

## Requirements
See requirements.txt for full list of dependencies including:
- PyTorch 2.1.2
- TorchVision 0.16.2
- einops 0.7.0

## Contributing
Pull requests are welcome. Please ensure all tests pass before submitting.

## License
[MIT](LICENSE)
