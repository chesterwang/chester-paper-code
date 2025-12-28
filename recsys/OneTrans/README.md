


```Python
python onetrans.py


#结果如下

=== OneTrans Demo ===
Input shapes:
  Sequential tokens: torch.Size([8, 100])
  Non-sequential tokens: torch.Size([8, 20])
Pyramid schedule: [100, 50, 25, 12, 6, 3, 0]
Output shape: torch.Size([8, 20])
Sample outputs: tensor([-0.0569,  0.3436,  0.0477, -0.3249,  0.7877, -0.0758,  0.6135,  0.0400,
         0.9961, -0.2574,  0.9585,  0.6592,  0.1664, -0.6396, -0.1433,  0.7120,
        -0.7176, -0.0590, -0.0566,  0.2597])

Model parameters: 60,954,625

Model structure:
  - Input tokens: 100 sequential + 20 non-sequential
  - Model dimensions: 256
  - Number of layers: 6
  - Attention heads: 4
  - Pyramid schedule: [100, 50, 25, 12, 6, 3, 0]
```