# 法律QA模型对比测试 - 本地数据/权重版
from pathlib import Path
import json
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("法律QA模型对比测试 - 本地数据/权重版")
print("=" * 70)

# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parent
BASE_MODEL = BASE_DIR / "models" / "base"
LORA_MODEL = BASE_DIR / "models" / "law-qa-qwen-lora"
DATA_FILE = BASE_DIR / "data" / "test-data.jsonl"
NUM_SAMPLES = 5  # 测试样本数量


def load_jsonl(path, limit=None):
    """Load line-delimited JSON, optionally sampling a subset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if limit is not None and len(records) > limit:
        random.seed(42)
        records = random.sample(records, limit)
    return records


# ==================== 加载测试数据 ====================
print("\n📖 加载测试数据...")
data = load_jsonl(DATA_FILE)
print(f"✅ 数据集大小: {len(data):,} 条")

# 提取问题和参考答案（适配 input/output 字段）
test_cases = []
for sample in data:
    question = sample.get("input")
    reference = sample.get("output")
    if question and reference:
        test_cases.append(
            {
                "question": question,
                "reference": reference,
                "source": sample.get("id", "unknown"),
            }
        )

# 随机抽取样本
random.seed(42)
test_cases = random.sample(test_cases, min(NUM_SAMPLES, len(test_cases)))

print(f"✅ 有效测试用例: {len(test_cases)} 个\n")

if not test_cases:
    raise SystemExit("没有可用的测试用例，请检查数据文件。")

# ==================== 加载模型 ====================
print("⏳ 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

finetuned_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

finetuned = PeftModel.from_pretrained(finetuned_base, LORA_MODEL)

print("✅ 加载完成\n")


# ==================== 相似度计算函数 ====================
def calculate_similarity(generated, reference):
    """计算生成答案与参考答案的相似度"""

    # 提取中文词汇（2字及以上）
    gen_words = set(re.findall(r"[\u4e00-\u9fff]{2,}", generated))
    ref_words = set(re.findall(r"[\u4e00-\u9fff]{2,}", reference))

    # 词汇重叠率
    if len(ref_words) > 0:
        common = gen_words & ref_words
        word_overlap = len(common) / len(ref_words)
        common_count = len(common)
    else:
        word_overlap = 0
        common_count = 0

    # 关键短语覆盖（4字及以上）
    ref_phrases = set(re.findall(r"[\u4e00-\u9fff]{4,}", reference))
    if len(ref_phrases) > 0:
        phrase_hits = sum(1 for phrase in ref_phrases if phrase in generated)
        phrase_coverage = phrase_hits / len(ref_phrases)
    else:
        phrase_coverage = 0

    # 综合得分
    score = (word_overlap * 0.6 + phrase_coverage * 0.4) * 100

    return {
        "score": score,
        "word_overlap": word_overlap,
        "phrase_coverage": phrase_coverage,
        "common_words": common_count,
        "total_ref_words": len(ref_words),
    }


# ==================== 对比测试 ====================
results = []

for i, test in enumerate(test_cases, 1):
    print(f"{'=' * 70}")
    print(f"测试 {i}/{len(test_cases)}")
    print(f"{'=' * 70}")
    print(f"来源: {test['source']}")

    print("\n【问题】")
    print(test["question"])

    print("\n【参考答案】（前200字）")
    ref_preview = (
        test["reference"][:200] + "..."
        if len(test["reference"]) > 200
        else test["reference"]
    )
    print(ref_preview)

    # 准备输入
    messages = [
        {"role": "system", "content": "你是一个专业的法律咨询助手。"},
        {"role": "user", "content": test["question"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(base.device)

    # 基座模型生成
    print("\n【基座模型回答】")
    print("-" * 70)
    with torch.no_grad():
        out = base.generate(**inputs, max_new_tokens=200, temperature=0.7)
    base_response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(base_response)

    # 计算基座模型相似度
    base_sim = calculate_similarity(base_response, test["reference"])
    print("\n📊 与参考答案的相似度:")
    print(f"  • 综合得分: {base_sim['score']:.1f}/100")
    print(
        f"  • 词汇重叠: {base_sim['word_overlap'] * 100:.1f}% ({base_sim['common_words']}/{base_sim['total_ref_words']})"
    )
    print(f"  • 短语覆盖: {base_sim['phrase_coverage'] * 100:.1f}%")

    # 微调模型生成
    print("\n【微调模型回答】")
    print("-" * 70)
    with torch.no_grad():
        out = finetuned.generate(**inputs, max_new_tokens=200, temperature=0.7)
    ft_response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(ft_response)

    # 计算微调模型相似度
    ft_sim = calculate_similarity(ft_response, test["reference"])
    print("\n📊 与参考答案的相似度:")
    print(f"  • 综合得分: {ft_sim['score']:.1f}/100")
    print(
        f"  • 词汇重叠: {ft_sim['word_overlap'] * 100:.1f}% ({ft_sim['common_words']}/{ft_sim['total_ref_words']})"
    )
    print(f"  • 短语覆盖: {ft_sim['phrase_coverage'] * 100:.1f}%")

    # 对比结果
    improvement = ft_sim["score"] - base_sim["score"]

    print(f"\n{'🎯 对比结果':=^70}")
    if improvement > 15:
        verdict = f"🏆 微调模型显著更好！提升 {improvement:.1f} 分"
    elif improvement > 10:
        verdict = f"✅ 微调模型明显更好，提升 {improvement:.1f} 分"
    elif improvement > 5:
        verdict = f"👍 微调模型更好，提升 {improvement:.1f} 分"
    elif improvement > 0:
        verdict = f"✅ 微调模型略好，提升 {improvement:.1f} 分"
    elif improvement > -5:
        verdict = f"🤝 两者接近，差距 {abs(improvement):.1f} 分"
    else:
        verdict = f"⚠️ 基座模型更好，差距 {abs(improvement):.1f} 分"

    print(verdict)
    print("=" * 70)
    print()

    results.append(
        {
            "base_score": base_sim["score"],
            "ft_score": ft_sim["score"],
            "improvement": improvement,
        }
    )

# ==================== 综合评估报告 ====================
print("\n")
print("=" * 70)
print("📊 综合评估报告")
print("=" * 70)

avg_base = sum(r["base_score"] for r in results) / len(results)
avg_ft = sum(r["ft_score"] for r in results) / len(results)
avg_improvement = sum(r["improvement"] for r in results) / len(results)

print("\n【平均相似度得分】")
print(f"  基座模型: {avg_base:.1f}/100")
print(f"  微调模型: {avg_ft:.1f}/100")
print(
    f"  平均提升: {avg_improvement:+.1f} 分 ({(avg_improvement / avg_base) * 100:+.1f}%)"
)

# 胜负统计
wins = sum(1 for r in results if r["improvement"] > 5)
draws = sum(1 for r in results if -5 <= r["improvement"] <= 5)
losses = sum(1 for r in results if r["improvement"] < -5)

print("\n【对战成绩】")
print(f"  微调明显更好: {wins}/{len(results)} ({wins / len(results) * 100:.0f}%)")
print(f"  两者接近: {draws}/{len(results)} ({draws / len(results) * 100:.0f}%)")
print(f"  基座更好: {losses}/{len(results)} ({losses / len(results) * 100:.0f}%)")

# 结论
print(f"\n{'📝 最终结论':=^70}")

if avg_improvement > 15:
    grade = "A+ (优秀)"
    conclusion = "✅ 微调效果显著！模型在训练数据上的表现远超基座模型。"
elif avg_improvement > 10:
    grade = "A (良好)"
    conclusion = "✅ 微调效果明显，模型明显优于基座模型。"
elif avg_improvement > 5:
    grade = "B+ (合格)"
    conclusion = "✅ 微调有效，模型优于基座模型。"
elif avg_improvement > 0:
    grade = "B (一般)"
    conclusion = "⚠️ 微调效果有限，提升不够明显。"
else:
    grade = "C (需改进)"
    conclusion = "⚠️ 微调效果不明显，需要检查训练过程。"

print(f"\n{conclusion}")
print(f"\n微调效果评级: {grade}")

if avg_improvement < 10:
    print("\n💡 改进建议:")
    print("  • 增加训练轮数（1 → 2-3 Epochs）")
    print("  • 增大 LoRA rank（r=4 → r=8）")
    print("  • 调整学习率")
    print("  • 检查数据质量")

print("\n" + "=" * 70)
print("✅ 测试完成")
print("=" * 70)
