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
# 该命令包含运行测试集，获取 benchmark
make test
```

## 本地问答

```bash
# 该命令会起一个本地网页，访问 localhost:3000 可以直接问答
make dev
```
