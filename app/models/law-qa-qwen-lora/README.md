# 法律QA模型 - Qwen2.5-1.5B LoRA

## 模型信息
- 基座模型: Qwen/Qwen2.5-1.5B-Instruct
- 微调方法: LoRA
- 训练数据: 4.9万条法律问答
- 训练效果: A+ 级别
- 创建日期: 2025-12-08

## 使用方法
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "/kaggle/input/law-qa-qwen-lora"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(base, MODEL_PATH)## 许可证
CC BY-NC 4.0
