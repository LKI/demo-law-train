# SFT Project: 微调大语言模型

- Base Model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Dev Environment: Linux or MacOS
- Training Data: https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT


## 安装依赖

```bash
# 该命令包含下载模型、下载数据集
# 如果遇到 private repo 或限流，请先 export HF_TOKEN=your_token
make install
```

## 代码格式化

```bash
# 使用 ruff 格式化代码
make fmt
```

## 模型测试

```bash
# 快速运行模型推理测试
make test
```

## 模型评估 (Benchmark)

```bash
# 在 SFT 数据集上运行完整评估 (Rouge-L)
make benchmark
```

## 本地问答

```bash
# 该命令会起一个本地网页，访问 localhost:3000 可以直接问答
make dev
```
