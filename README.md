# SFT Project: 微调大语言模型

- Base Model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Dev Environment: Linux or MacOS
- Training Data: https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT


## 安装依赖

```bash
# 该命令包含下载模型、下载数据集
make install
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
