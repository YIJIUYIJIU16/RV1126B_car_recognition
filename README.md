# RV1126B Car Recognition

基于 Rockchip RKNN 的 RV1126B 车牌与车辆颜色识别 C++ 部署示例。

本项目参考并改写自 [we0091234/Car_recognition](https://github.com/we0091234/Car_recognition)。原项目主要提供 Python/PyTorch 训练与推理流程；本仓库将推理链路整理为可在 RV1126B 板端运行的 C++ 工程，并补充了 ONNX/RKNN 转换脚本、量化数据集生成脚本、图片批量推理和 V4L2 摄像头流推理入口。

## 功能

- 车牌/车辆检测：检测单层车牌、双层车牌和车辆目标。
- 车牌识别：识别车牌字符序列，并输出车牌颜色。
- 车辆颜色识别：对车辆 ROI 输出 11 类颜色结果。
- RKNN 部署：提供 detect、plate_rec_color、car_recognition 三个模型的转换脚本。
- C++ 推理：提供单图、图片目录批量推理和板端 V4L2 摄像头流推理。

## 项目来源与说明

- 上游参考仓库：[https://github.com/we0091234/Car_recognition](https://github.com/we0091234/Car_recognition)
- 本仓库依赖 RKNN Model Zoo 的构建脚本、3rdparty 和 utils 目录。
- `Car_recognition-master/` 保留了上游 Python 工程，便于对照模型结构、训练脚本和原始推理流程。
- `cpp/` 是本项目主要改写内容，负责在板端加载 RKNN 模型并执行完整识别流水线。

如果继续发布或二次开发，请保留对上游项目和 RKNN Model Zoo 的引用说明。

## 目录结构

```text
car_recognition/
├── README.md
├── Car_recognition-master/          # 上游 Python/PyTorch 项目副本，供训练、权重来源和逻辑对照
├── cpp/
│   ├── CMakeLists.txt               # C++ 构建配置，生成两个板端 demo
│   ├── car_recognition.h            # 公共结构体、类别表、模型接口声明
│   ├── detect.cc                    # 车牌/车辆检测 RKNN 初始化、前处理、后处理
│   ├── plate_rec.cc                 # 车牌字符与车牌颜色识别
│   ├── car_color.cc                 # 车辆颜色识别
│   ├── pipeline.cc                  # 三模型串联流水线，供图片和视频流入口复用
│   ├── main.cc                      # 图片/目录推理入口
│   └── stream_demo.cc               # V4L2 摄像头流推理入口
├── model/
│   ├── detect.onnx                  # 检测模型 ONNX
│   ├── detect.rknn                  # 检测模型 RKNN
│   ├── plate_rec_color.onnx         # 车牌字符+颜色识别 ONNX
│   ├── plate_rec_color.rknn         # 车牌字符+颜色识别 RKNN
│   ├── car_recognition.onnx         # 车辆颜色识别 ONNX
│   ├── car_recognition.rknn         # 车辆颜色识别 RKNN
│   ├── dataset.txt                  # 默认量化图片列表
│   ├── single_blue.jpg              # 测试图片
│   ├── calibration_sets/            # 可选，脚本生成的专用量化集
│   └── platech_glyphs/              # 中文绘制所需字形贴图
└── python/
    ├── export_detect_onnx.py        # detect.pt 导出 detect.onnx
    ├── export_plate_rec_color_onnx.py
    ├── export_onnx.py               # car_rec_color.pth 导出 car_rec_color.onnx
    ├── convert_detect.py            # detect.onnx 转 detect.rknn
    ├── convert_plate_rec_color.py   # plate_rec_color.onnx 转 plate_rec_color.rknn
    ├── convert.py                   # car_rec_color.onnx 转 car_recognition.rknn
    ├── generate_calibration_sets.py # 从标注数据生成三类模型各自的量化集
    └── generate_platech_glyphs.py   # 从 platech.ttf 生成 C++ 绘制用中文字形贴图
```

## 环境准备

PC 端用于导出和转换模型：

- Python 3.8+
- PyTorch
- onnx
- opencv-python
- rknn-toolkit2 或与你 SDK 匹配的 RKNN 转换工具

板端/交叉编译环境：

- RV1126B Linux SDK
- RKNN runtime
- aarch64-linux-gnu 交叉编译器
- RKNN Model Zoo 的 `3rdparty/`、`utils/` 和 `build-linux.sh`

摄像头流 demo 还依赖：

- `RkYoloRtspServer-master/thirdparty/libv4l2cc`

当前 `cpp/CMakeLists.txt` 中的相对路径为：

```cmake
../../../../RkYoloRtspServer-master/thirdparty/libv4l2cc
```

如果你的目录不同，需要同步修改该路径。

## 模型说明

本工程使用三个模型串联：

| 模型 | 默认文件 | 输入尺寸 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| 车牌/车辆检测 | `model/detect.rknn` | `1x3x384x384` | bbox、关键点、类别分数 | 检测单层车牌、双层车牌、车辆 |
| 车牌识别+颜色 | `model/plate_rec_color.rknn` | `1x3x48x168` | 字符序列、车牌颜色 | 识别车牌号码和颜色 |
| 车辆颜色 | `model/car_recognition.rknn` | `1x3x64x64` | 11 类颜色概率 | 识别车辆颜色 |

检测类别：

```text
0: single_plate
1: double_plate
2: car
```

车牌颜色类别：

```text
black, blue, green, white, yellow
```

车辆颜色类别：

```text
black, blue, yellow, brown, green, gray, orange, pink, purple, red, white
```

## ONNX 导出

默认导出脚本会从上游工程权重目录读取 PyTorch 权重：

```text
../../../../Car_recognition/weights/
```

也就是说，从 `examples/car_recognition/python/` 往上四级后，需要存在一个同级的 `Car_recognition/weights/` 目录。你也可以直接修改三个导出脚本里的 `MODEL_PATH`，指向自己的权重文件。

导出命令：

```bash
cd examples/car_recognition/python

python3 export_detect_onnx.py
python3 export_plate_rec_color_onnx.py
python3 export_onnx.py
```

输出文件：

```text
../model/detect.onnx
../model/plate_rec_color.onnx
../model/car_rec_color.onnx
```

如果你希望最终文件名统一为 `car_recognition.onnx`，可以在导出后复制或重命名：

```bash
cp ../model/car_rec_color.onnx ../model/car_recognition.onnx
```

## RKNN 转换

RV1126B 推荐使用 int8 量化：

```bash
cd examples/car_recognition/python

python3 convert_detect.py \
  ../model/detect.onnx rv1126b i8 ../model/detect.rknn \
  ../model/dataset.txt

python3 convert_plate_rec_color.py \
  ../model/plate_rec_color.onnx rv1126b i8 ../model/plate_rec_color.rknn \
  ../model/dataset.txt

python3 convert.py \
  ../model/car_rec_color.onnx rv1126b i8 ../model/car_recognition.rknn \
  ../model/dataset.txt
```

转换脚本参数格式：

```text
python3 convert_xxx.py onnx_model_path platform dtype output_rknn_path dataset_path
```

- `platform`：目标芯片，例如 `rv1126b`。
- `dtype`：`i8`、`u8` 或 `fp`，RV1126B 常用 `i8`。
- `output_rknn_path`：可选，RKNN 输出路径。
- `dataset_path`：可选，量化图片列表。

## 量化集生成

为了减少三个模型之间量化输入分布不一致的问题，可以从标注数据生成各自专用的量化集：

```bash
cd examples/car_recognition/python

python3 generate_calibration_sets.py \
  --images /path/to/images \
  --labels /path/to/labels \
  --output-root ../model/calibration_sets \
  --max-detect 300 \
  --max-plate 500 \
  --max-car 500 \
  --split-double-plate
```

标注格式沿用上游 YOLO 格式：

```text
label x y w h pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
```

- `0`：单层车牌
- `1`：双层车牌
- `2`：车辆
- 车牌关键点顺序：左上、右上、右下、左下
- 车辆类别不需要关键点，可填 `-1`

生成结果：

```text
model/calibration_sets/detect_dataset.txt
model/calibration_sets/plate_rec_dataset.txt
model/calibration_sets/car_color_dataset.txt
```

使用专用量化集转换：

```bash
cd examples/car_recognition/python

python3 convert_detect.py \
  ../model/detect.onnx rv1126b i8 ../model/detect.rknn \
  ../model/calibration_sets/detect_dataset.txt

python3 convert_plate_rec_color.py \
  ../model/plate_rec_color.onnx rv1126b i8 ../model/plate_rec_color.rknn \
  ../model/calibration_sets/plate_rec_dataset.txt

python3 convert.py \
  ../model/car_rec_color.onnx rv1126b i8 ../model/car_recognition.rknn \
  ../model/calibration_sets/car_color_dataset.txt
```

建议样本数量：

- `detect`：100-300 张真实场景图。
- `plate_rec_color`：200-500 张矫正后的车牌裁剪图。
- `car_recognition`：200-500 张车辆 ROI 裁剪图。

## C++ 编译

在 RKNN Model Zoo 根目录编译：

```bash
cd /data1/CYC/RV/rknn_model_zoo-main
./build-linux.sh -t rv1126b -a aarch64 -d car_recognition
```

如果交叉编译器不在 `PATH` 中，先设置 `GCC_COMPILER`：

```bash
export GCC_COMPILER=/data1/CYC/RV/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

cd /data1/CYC/RV/rknn_model_zoo-main
./build-linux.sh -t rv1126b -a aarch64 -d car_recognition
```

生成目录：

```text
install/rv1126b_linux_aarch64/rknn_car_recognition_demo/
```

生成程序：

```text
rknn_car_recognition_demo
rknn_car_recognition_stream_demo
```

安装目录会同时包含 `model/`，用于板端直接运行。

## 图片推理

进入板端安装目录：

```bash
cd /data1/CYC/RV/rknn_model_zoo-main/install/rv1126b_linux_aarch64/rknn_car_recognition_demo
```

单张图片：

```bash
./rknn_car_recognition_demo \
  -d model/detect.rknn \
  -p model/plate_rec_color.rknn \
  -c model/car_recognition.rknn \
  -i model/single_blue.jpg \
  -o out.jpg
```

图片目录批量推理：

```bash
./rknn_car_recognition_demo \
  -d model/detect.rknn \
  -p model/plate_rec_color.rknn \
  -c model/car_recognition.rknn \
  -i /home/workspace/test_images \
  -o /home/workspace/test_results
```

参数说明：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-d`, `--detect_model` | 检测 RKNN 模型路径 | `model/detect.rknn` |
| `-p`, `--plate_model` | 车牌识别 RKNN 模型路径 | `model/plate_rec_color.rknn` |
| `-c`, `--car_model` | 车辆颜色 RKNN 模型路径 | `model/car_recognition.rknn` |
| `-i`, `--image` | 输入图片或图片目录 | 必填 |
| `-o`, `--output` | 输出图片或输出目录 | `result.jpg` |

## 摄像头流推理

板端 V4L2 摄像头输入示例：

```bash
cd /data1/CYC/RV/rknn_model_zoo-main/install/rv1126b_linux_aarch64/rknn_car_recognition_demo

./rknn_car_recognition_stream_demo \
  -d model/detect.rknn \
  -p model/plate_rec_color.rknn \
  -c model/car_recognition.rknn \
  -D /dev/video0 \
  -W 640 \
  -H 480 \
  -f 30 \
  -m 100
```

参数说明：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-D`, `--device` | V4L2 设备路径 | `/dev/video0` |
| `-W`, `--width` | 采集宽度 | `640` |
| `-H`, `--height` | 采集高度 | `480` |
| `-f`, `--fps` | 采集帧率 | `30` |
| `-m`, `--max_frames` | 最多处理帧数，`0` 表示持续运行 | `0` |

当前流式 demo 只负责读取本地 V4L2 摄像头、执行推理并打印每帧耗时；暂未实现 RTSP 输入、RTSP 输出或视频编码推流。

## C++ 推理流程

1. `detect.rknn` 对整图执行检测，得到车牌/车辆框和车牌四点关键点。
2. 如果目标是车辆，裁剪车辆 ROI，送入 `car_recognition.rknn` 输出车辆颜色。
3. 如果目标是车牌，根据四点关键点做透视变换。
4. 双层车牌会先按上下层切分再拼接为单行图。
5. `plate_rec_color.rknn` 输出车牌字符序列和车牌颜色。
6. `draw_results` 在图像上绘制识别结果。

## 常见问题

### 1. 导出脚本找不到权重

检查 `python/export_*.py` 中的 `MODEL_PATH`。默认路径指向：

```text
../../../../Car_recognition/weights/
```

如果你的上游工程目录叫 `Car_recognition-master` 或放在其他位置，需要修改脚本路径。

### 2. 编译找不到 libv4l2cc

检查 `cpp/CMakeLists.txt` 中的：

```cmake
../../../../RkYoloRtspServer-master/thirdparty/libv4l2cc
```

将它改成你本地 `libv4l2cc` 的真实路径，或者临时移除 `stream_demo.cc` 相关目标，只编译图片推理 demo。

### 3. 板端中文显示异常

检查 `model/platech_glyphs/` 是否随安装目录一起复制。必要时重新生成字形贴图：

```bash
cd examples/car_recognition/python
python3 generate_platech_glyphs.py \
  --font /path/to/platech.ttf \
  --output-dir ../model/platech_glyphs
```

### 4. 量化后识别效果下降

优先使用真实部署场景数据生成三份专用量化集，避免检测图、车牌裁剪图、车辆裁剪图混用同一份输入分布。

## 致谢

- [we0091234/Car_recognition](https://github.com/we0091234/Car_recognition)
- [Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [yolov5-face](https://github.com/deepcam-cn/yolov5-face)
- [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
