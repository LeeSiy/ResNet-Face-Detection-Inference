# ResNet-Face-Detection-Inference
Inference code for face detection

# Reference
- 1. model from https://github.com/HuangYG123/CurricularFace
- 2. model from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
- 3. model from https://github.com/deepinsight/insightface

# Inference Code file address
- /face.evoLVe.PyTorch/align/test_match3.py
- using detecting code from insightface
- ![1](https://user-images.githubusercontent.com/62841284/116346345-0a009a00-a825-11eb-9350-d661d8c37ad2.png)
- ![2](https://user-images.githubusercontent.com/62841284/116346351-0c62f400-a825-11eb-8c5a-ad01edb0fc16.png)

- use cv2.putText(group_box, 'Makron', (x1_box_list[num], y1_box_list[num]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) for labeling
- ![3](https://user-images.githubusercontent.com/62841284/116355011-353eb580-a834-11eb-945f-1b4951098818.png)


- /face.evoLVe.PyTorch/align/test_match.py

- ![Screenshot from 2021-04-26 11-02-18](https://user-images.githubusercontent.com/62841284/116019487-3da1bf80-a67f-11eb-9f40-ffa31554ffce.png)


- /face.evoLVe.PyTorch/align/test.py

- ![Screenshot from 2021-04-26 10-31-34](https://user-images.githubusercontent.com/62841284/116019244-c2d8a480-a67e-11eb-8863-eeea86a69cd8.png)

