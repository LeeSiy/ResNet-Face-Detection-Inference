# ResNet Face Detection Inference
Inference code for face detection

# Reference
- 1. https://github.com/HuangYG123/CurricularFace
- 2. https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
- 3. https://github.com/deepinsight/insightface

# Export library
- export PYTHONPATH=~/Face/face.evoLVe.PyTorch/util/:$PYTHONPATH
- export PYTHONPATH=~/Face/face.evoLVe.PyTorch/:$PYTHONPATH
- export PYTHONPATH=~/Face/insightface/:$PYTHONPATH
- export PYTHONPATH=~/Face/insightface/detection/:$PYTHONPATH
- sudo apt-get install google-perftools
- export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"

# Inference Code file address
- /face.evoLVe.PyTorch/test_video.py

- video example inference code with threads (example screen shot images are from https://www.pexels.com/
  - detecting thread 
  - classifying thread
  - reading video thread
  - extracting feature thread
  
  
  
  
  https://user-images.githubusercontent.com/62841284/116836364-36d7f700-ac01-11eb-9d32-4c033d59b27c.mp4


  ![r1](https://user-images.githubusercontent.com/62841284/116641523-d732df00-a9a7-11eb-9e7d-79bf48b5da00.png)
  
  -Label list
  
  ![r2](https://user-images.githubusercontent.com/62841284/116641538-dbf79300-a9a7-11eb-8624-f83451e9f876.png)


- /face.evoLVe.PyTorch/align/test_match3.py
- using detecting code from insightface
  
  ![1](https://user-images.githubusercontent.com/62841284/116346345-0a009a00-a825-11eb-9350-d661d8c37ad2.png)
  ![2](https://user-images.githubusercontent.com/62841284/116346351-0c62f400-a825-11eb-8c5a-ad01edb0fc16.png)

- use cv2.putText(group_box, 'Makron', (x1_box_list[num], y1_box_list[num]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) for labeling
  ![3](https://user-images.githubusercontent.com/62841284/116355011-353eb580-a834-11eb-945f-1b4951098818.png)


- /face.evoLVe.PyTorch/align/test_match.py

  ![Screenshot from 2021-04-26 11-02-18](https://user-images.githubusercontent.com/62841284/116019487-3da1bf80-a67f-11eb-9f40-ffa31554ffce.png)


- /face.evoLVe.PyTorch/align/test.py

  ![Screenshot from 2021-04-26 10-31-34](https://user-images.githubusercontent.com/62841284/116019244-c2d8a480-a67e-11eb-8863-eeea86a69cd8.png)

