# 语音识别

同济大学 2022级 计算机科学与技术学院 软件工程专业 机器智能方向 语音识别课程作业

授课教师：沈莹

授课学期：2024-2025年度 秋季学期

# 任务: DeepSpeech

基于MindSpore框架搭建deepspeech2语音识别模型

说明实验重要步骤的产出以及评估结果

回答问题：

仅训练一个历元就要花费 50 个小时，如何加快程序运行速度？

# 实验基本介绍

## 实验背景

DeepSpeech2 是由百度研究院提出的一种端到端语音识别模型，基于CTC（Connectionist Temporal Classification）损失函数进行训练。与传统的手工设计管道不同，DeepSpeech2 使用神经网络直接从语音信号中学习特征，并生成对应的文本输出。该模型能够处理多种复杂的语音场景，包括嘈杂的环境、不同的口音以及多种语言。DeepSpeech2 的提出极大地简化了语音识别系统的开发流程，并显著提升了识别性能。

MindSpore 是华为推出的一款开源深度学习框架，旨在提供高效、灵活的开发体验。MindSpore 支持多种硬件平台，包括 CPU、GPU 和 Ascend，适用于从研究到生产的多种场景。通过 MindSpore，开发者可以快速搭建和训练深度学习模型，并实现高效的推理和部署。

## 实验目的

- 熟悉 MindSpore 训练网络的流程：通过实验，学习者将了解如何使用 MindSpore 进行模型的搭建、训练和评估，掌握 MindSpore 的基本使用方法。

- 熟悉 MindSpore 搭建 DeepSpeech2 网络的训练和评估过程：学习者将深入了解 DeepSpeech2 模型的架构设计，并通过 MindSpore 实现该模型的训练和评估，掌握如何使用 MindSpore 进行端到端语音识别任务。

- 了解 Linux 操作命令：实验平台基于 Linux 环境，学习者将通过实验熟悉常用的 Linux 操作命令，提升在 Linux 系统下的开发能力。

## 实验开发环境

- 实验平台：ECS（云服务器）

- 框架：MindSpore 1.6

- 硬件：CPU

- 软件环境：Python 3.9.0、MindSpore 1.6

# 实验过程

## 环境准备

购买ECS资源并进行远程登陆，以创建开发环境。

## 代码及数据下载

`deepspeech2` 项目代码位于 `models/offical/audio/deepspeech2` 文件夹下。

训练和推理的相关参数在config.py文件中。

数据集为LibriSpeech数据集。
- 训练集：train-clean-100: [6.3G] (100小时的无噪音演讲训练集)
- 验证集：dev-clean.tar.gz [337M] (无噪音)、dev-other.tar.gz [314M] (有噪音)
- 测试集：test-clean.tar.gz [346M] (测试集, 无噪音)、test-other.tar.gz [328M] (测试集, 有噪音)

## 数据预处理

### 准备工作

1. 安装 `Python 3.9.0`
2. 安装 `MindSpore` 和所需要的依赖包
3. 下载数据预处理 `SeanNaren` 脚本

### LibriSpeech数据预处理

1. 上传本地数据集到 MobaXterm 服务器上并修改路径
2. 执行数据集处理命令，执行命令如下
    ```bash
    python librispeech.py
    ```
3. `json` 文件转 `csv` 文件：在 `deepspeech.pytorch` 目录下创建 `json_to_csv.py`，并把代码复制到文件。

## 模型训练与评估

### 模型训练

1. 在 `DeepSpeech2` 目录下创建 `deepspeech_pytorch` 目录，并在 `deepspeech_pytorch` 目录下创建 `decoder.py` 文件
2. 修改 `src` 下的 `config.py`：
    - 修改batch_size 为1（一次处理数据量大小，该数据与服务器设备性能相关）
    - 修改epochs为1（大概用时48h，可根据实际需求调整）
    - 修改train_manifest为libri_train_manifest.csv实际路径
    - 修改test_manifest为libri_test_clean_manifest.csv为实际路径
    - 修改eval_config的加窗类型把hanning改为hann
3. 安装Python依赖、下载预训练模型，下载命令如下：
    ```bash
    wget https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com/ASR/DeepSpeech.ckpt
    ```
4. 修改训练启动脚本 `scripts` 目录下的 `run_standalone_train_cpu.sh`，加载预训练模型：
    ```shell
    PATH_CHECKPOINT=$1
    python ./train.py --device_target 'CPU' --pre_trained_model_path $PATH_CHECKPOINT
    ```
5. 在 `DeepSpeech2` 目录下进行模型训练，输入如下命令：
    ```bash
    bash scripts/run_standalone_train_cpu.sh PATH_CHECKPOINT
    # PATH_CHECKPOINT 预训练文件路径
    ```

    或在后台运行：

    ```bash
    nohup bash scripts/run_standalone_train_cpu.sh '/home/work/models/official/audio/DeepSpeech2/DeepSpeech.ckpt' > train.log 2>&1 &
    ```
6. 查看训练日志，当前目录下train.log：
    ```bash
    tail –f train.log
    ```
    ![](../DeepSpeech/assets/1.png)

### 模型评估

1. 模型评估：

    ```bash
    # CPU评估
    bash scripts/run_eval_cpu.sh [PATH_CHECKPOINT]
    # [PATH_CHECKPOINT] 模型checkpoint文件
    # 参考样例：
    bash scripts/run_eval_cpu.sh ./checkpoint/ DeepSpeech-1_140.ckpt
    ```

2. 查看评估日志：

    ```bash
    tail –f eval.log
    ```

    ![](../DeepSpeech/assets/2.png)

3. 模型导出，修改 `export.py` 的代码：

    ```bash
    config = train_config
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
    with open(config.DataConfig.labels_path) as label_file:
    labels = json.load(label_file)
    ```

4. 转换并导出模型文件：

    ```bash
    python export.py --pre_trained_model_path  ./ checkpoint/ DeepSpeech-1_856.ckpt
    ```

# 思考题

## 硬件优化

使用更高性能的硬件，如 GPU（NVIDIA V100/A100）或 AI 专用芯片（如 Ascend 910）。CPU 的性能有限，特别是在深度学习任务中，GPU 和专用芯片可以显著缩短训练时间。在华为云中，根据预算选择更强大的计算实例。

## 软件和算法优化

1. 调整 `Batch Size`：

    - 增大 `batch_size`，一次处理更多数据，减少梯度计算的次数。
    - 需要平衡批量大小与内存消耗，避免显存溢出。

2. 优化数据加载：

    - 使用多线程或异步数据加载（如 MindSpore 的 DataLoader 提供的并行功能）。
    - 确保数据预处理（如数据增强、解码）不会成为训练的瓶颈。

3. 混合精度训练：

    - 启用混合精度（FP16 和 FP32 结合），利用硬件支持更快的浮点运算。
    - MindSpore 提供 `mindspore.amp` 模块支持混合精度训练。

4. 使用增量学习：

    - 从预训练模型（如 DeepSpeech2 的开源预训练权重）开始训练，只更新后几层权重，减少计算量。

5. 模型剪枝和蒸馏：

    - 剪掉冗余的网络权重（模型剪枝）。
    - 使用更小的学生模型通过知识蒸馏方法学习。

6. 调整优化器和学习率：

    - 使用更快收敛的优化器（如 Adam、LAMB 等）。
    - 设置动态学习率调度器，根据训练进程调整学习率。

## 减少数据量

1. 精简数据集：

    - 使用数据子集（如 train-clean-100 数据集），通过迁移学习提升小数据集的效果。
    - 数据扩增：在训练集上进行数据增强（如时间变换、音量调整），以少量数据实现更好的泛化。

2. 减少 Epoch 或早停策略：

    - 调整训练策略，减少不必要的历元（通过验证集性能早停）。