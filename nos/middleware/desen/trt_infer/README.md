# PP-Picodet-OnnxInfer 

**该分支为基于PP-Picodet(L416)模型的数据脱敏。**

### 模型训练及结果

数据脱敏模型用于在车端对车牌和人脸做脱敏处理。数据集包括行车driving数据集和驻车parking鱼眼数据集。
上一版本的数据脱敏模型基于CenterNet(ResNet18)做训练，精度和推理速度均有待提高。本版本使用PP-Picodet(L416)模型进行训练，并将driving和parking数据集合并。精度对比如下：

|       数据集测试精度         | driving数据集   |parking数据集  | combined数据集 |
|--------------------|:--------:|:--------:|:-------:|
|Picodet(L416)| 0.229| 0.189   | 0.209 |
|CenterNet(ResNet18) | 0.219      |  0.118    |    |

如图所示，PP-Picodet(L416)模型在driving和parking数据集上的测试精度均高于原CenterNet模型。此外，经测试，PP-Picodet(L416)的Onnx模型推理速度为CenterNet模型的15.4倍。

### Onnx推理

将PP-Picodet(L416)模型转换为Onnx模型时，去除了模型的NMS等后处理步骤。去除后处理的Onnx模型保存路径为weights/picodet_l_416_nonms.onnx. 使用Onnx做推理——

```sh
python infer.py --imgpath images/sample.jpg --modelpath weights/picodet_l_416_nonms.onnx \
    --classfile hozon.names  --confThreshold 0.025 --nmsThreshold 0.6 --infer_config infer_cfg.yml
```

其中，--imgpath为推理图片路径，--modelpath为onnx路径，--classfile为标签名称，--confThreshold为置信度阈值，--nmsThreshold为NMS时的IOU阈值，--infer_config为模型参数文件，根据PaddlePaddle官方代码将PP-Picodet转换为Onnx训练后会自动生成。

### OM文件生成
将Onnx转换为OM格式，用于在MDC上执行推理。OM文件存储路径为weights/picodet_l_416_nonms.om

### TensorRT推理

#### 1. 配置环境docker环境

```shell
# 下载docker
docker pull nvcr.io/nvidia/tensorrt:23.10-py3
# 新建docker容器，其中docker id请用docker images 查看下载好的image的id后替换
# -v参数挂载硬盘，根据自己实际情况修改
# nvidia-
# docker run --name trt -it -v /home/hozon/data/:/data 130a238396e7 /bin/bash
docker run --name trt -it -v /home/lbl/Downloads/Picodet_OnnxInfer/:/data 130a238396e7 /bin/bash
docker start trt
docker exec -ti trt /bin/bash
```

#### 2. 安装相应依赖库

```shell
apt-get update && apt-get install libgl1 
pip install opencv-python
pip install pyyaml
pip install onnxruntime
pip install onnx
```

#### 3. Python执行

```shell
# 进入项目根目录后
# onnx 执行推理, 默认参数可以参考脚本中args
python infer.py --imgpath images/sample.jpg --modelpath weights/picodet_l_416_nonms.onnx --classfile hozon.names --confThreshold 0.13 --nmsThreshold 0.6 --infer_config infer_cfg.yml
# tensorrt 推理，默认参数可以参考脚本中args, 模型文件可以用.onnx模型也可以用.plan
python infer_trt.py --imgpath images/sample.jpg --modelpath weights/picodet_l_416_nonms.plan --classfile hozon.names --confThreshold 0.13 --nmsThreshold 0.6 --infer_config infer_cfg.yml
```

#### 4. C++ 编译运行

```shell
# 进入项目根目录后
cd trt_c++
mkdir build
cd build
cmake ..
# 注意 .plan模型需要根据不同的硬件进行生成，此版本支持A100
./bin/de_privacy_infer ../images/sample.jpg ../weights/picodet_l_416_nonms.plan
```

#### 5. TRT .plan模型生成

```shell
# 进入项目根目录后
python infer_trt.py --imgpath images/sample.jpg --modelpath weights/picodet_l_416_nonms.onnx --classfile hozon.names --confThreshold 0.13 --nmsThreshold 0.6 --infer_config infer_cfg.yml --save_engine
```

运行完成后，生成`engine.plan`文件，自行重命名即可。
