以包的形式打包并安装此库：推理
或者以源码的形式安装此库：微调 + 推理

1. python打包此库 
```python
python setup.py bdist_wheel
```
在生成的dist文件夹中会出现.whl文件

安装.whl文件
```python
pip install dist/xxxx.whl
```

2. 以源码形式安装
```python
pip install -e .
```

3. segment-anything的外加内容

- segment_anything/inference.py
整图推理一张图片
- segment_anything/interactive_inference.py
交互式推理
- segment_anything/lightning_modeling/
模型微调(需要以源码形式安装)
```python
python segment_anything/lightning_modeling/my_engine.py
```

TODO
pytorch lightning框架微调的完善

[ ] 评估指标和loss的记录  
[ ] 完善engine中自动保存权重  