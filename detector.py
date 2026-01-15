from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
import re
from collections import Counter

from url_extractor import is_url, extract_text

# ==================== 模型配置 ====================
# 1. GPT通用检测器 (RoBERTa-large)
GPT_DETECTOR_MODEL = "openai-community/roberta-large-openai-detector"

# 2. 困惑度分析模型 (GPT-2)
PERPLEXITY_MODEL = "gpt2"

# 3. ChatGPT中文检测器
CHATGPT_ZH_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta-chinese"

# 4. RoBERTa base检测器
ROBERTA_BASE_MODEL = "openai-community/roberta-base-openai-detector"

# 5. 学术文本检测器
ACADEMIC_MODEL = "andreas122001/roberta-academic-detector"

# ==================== 权重配置 ====================
WEIGHTS = {
    "gpt_detector": 0.20,      # GPT通用检测器 (large)
    "perplexity": 0.15,        # 困惑度分析
    "chatgpt_zh": 0.20,        # ChatGPT中文检测器
    "roberta_base": 0.15,      # RoBERTa base检测器
    "academic": 0.10,          # 学术文本检测器
    "burstiness": 0.10,        # 突发性分析
    "vocab_richness": 0.10,    # 词汇丰富度
}

# ==================== 模型加载 ====================
print("正在加载模型...")

# 1. GPT检测器
gpt_tokenizer = AutoTokenizer.from_pretrained(GPT_DETECTOR_MODEL)
gpt_model = AutoModelForSequenceClassification.from_pretrained(GPT_DETECTOR_MODEL)

# 2. GPT-2 困惑度模型
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(PERPLEXITY_MODEL)
gpt2_model = GPT2LMHeadModel.from_pretrained(PERPLEXITY_MODEL)
gpt2_model.eval()

# 3. ChatGPT中文检测器
chatgpt_zh_tokenizer = AutoTokenizer.from_pretrained(CHATGPT_ZH_MODEL)
chatgpt_zh_model = AutoModelForSequenceClassification.from_pretrained(CHATGPT_ZH_MODEL)

# 4. RoBERTa base检测器
roberta_base_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE_MODEL)
roberta_base_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_BASE_MODEL)

# 5. 学术文本检测器
academic_tokenizer = AutoTokenizer.from_pretrained(ACADEMIC_MODEL)
academic_model = AutoModelForSequenceClassification.from_pretrained(ACADEMIC_MODEL)

print("模型加载完成！")


# ==================== 检测函数 ====================

def detect_with_gpt_detector(text: str) -> float:
    """使用GPT通用检测器，返回AI生成概率"""
    inputs = gpt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = gpt_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        ai_prob = probs[0][1].item()
    return ai_prob


def calculate_perplexity(text: str) -> float:
    """使用GPT-2计算文本困惑度"""
    encodings = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = gpt2_model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())
    return perplexity


def perplexity_to_probability(perplexity: float) -> float:
    """将困惑度转换为AI生成概率（困惑度越低，AI概率越高）"""
    center = 45
    scale = 0.1
    prob = 1 / (1 + math.exp(scale * (perplexity - center)))
    return prob


def detect_with_chatgpt_zh(text: str) -> float:
    """使用ChatGPT中文检测器，返回AI生成概率"""
    inputs = chatgpt_zh_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = chatgpt_zh_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        ai_prob = probs[0][1].item()
    return ai_prob


def detect_with_roberta_base(text: str) -> float:
    """使用RoBERTa base检测器，返回AI生成概率"""
    inputs = roberta_base_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = roberta_base_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        ai_prob = probs[0][1].item()
    return ai_prob


def detect_with_academic(text: str) -> float:
    """使用学术文本检测器，返回AI生成概率"""
    inputs = academic_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = academic_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        # 检查输出维度
        if probs.shape[1] == 2:
            ai_prob = probs[0][1].item()
        else:
            ai_prob = probs[0][0].item()
    return ai_prob


def calculate_burstiness(text: str) -> float:
    """
    计算文本突发性（句子长度方差）
    AI生成的文本通常句子长度更均匀（低突发性）
    人类写的文本句子长度变化更大（高突发性）
    """
    sentences = re.split(r'[。！？.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    if len(sentences) < 2:
        return 0.5

    lengths = [len(s) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_len if mean_len > 0 else 0

    return cv


def burstiness_to_probability(burstiness: float) -> float:
    """将突发性转换为AI生成概率"""
    center = 0.45
    scale = 5
    prob = 1 / (1 + math.exp(scale * (burstiness - center)))
    return prob


def calculate_vocab_richness(text: str) -> float:
    """
    计算词汇丰富度 (Type-Token Ratio)
    AI生成的文本通常词汇更重复（低丰富度）
    人类写的文本词汇更多样化（高丰富度）
    """
    # 简单分词（按字符和标点分割）
    # 对于中文，每个字都算一个token
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text)

    if len(tokens) < 10:
        return 0.5

    # Type-Token Ratio (TTR)
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens)

    return ttr


def vocab_richness_to_probability(ttr: float) -> float:
    """
    将词汇丰富度转换为AI生成概率
    低TTR（< 0.4）-> 高AI概率（词汇重复多）
    高TTR（> 0.7）-> 低AI概率（词汇丰富）
    """
    center = 0.55
    scale = 8
    # TTR越低，AI概率越高
    prob = 1 / (1 + math.exp(scale * (ttr - center)))
    return prob


def detect(text: str) -> dict:
    """
    多维度检测文本是否为AI生成
    返回各维度分析结果和加权综合概率
    """
    results = {}

    # 1. GPT通用检测器
    results["gpt_detector"] = detect_with_gpt_detector(text)

    # 2. 困惑度分析
    perplexity = calculate_perplexity(text)
    results["perplexity"] = perplexity
    results["perplexity_prob"] = perplexity_to_probability(perplexity)

    # 3. ChatGPT中文检测器
    results["chatgpt_zh"] = detect_with_chatgpt_zh(text)

    # 4. RoBERTa base检测器
    results["roberta_base"] = detect_with_roberta_base(text)

    # 5. 学术文本检测器
    results["academic"] = detect_with_academic(text)

    # 6. 突发性分析
    burstiness = calculate_burstiness(text)
    results["burstiness"] = burstiness
    results["burstiness_prob"] = burstiness_to_probability(burstiness)

    # 7. 词汇丰富度
    vocab_richness = calculate_vocab_richness(text)
    results["vocab_richness"] = vocab_richness
    results["vocab_richness_prob"] = vocab_richness_to_probability(vocab_richness)

    # 加权计算综合概率
    combined_prob = (
        results["gpt_detector"] * WEIGHTS["gpt_detector"] +
        results["perplexity_prob"] * WEIGHTS["perplexity"] +
        results["chatgpt_zh"] * WEIGHTS["chatgpt_zh"] +
        results["roberta_base"] * WEIGHTS["roberta_base"] +
        results["academic"] * WEIGHTS["academic"] +
        results["burstiness_prob"] * WEIGHTS["burstiness"] +
        results["vocab_richness_prob"] * WEIGHTS["vocab_richness"]
    )

    results["probability"] = combined_prob
    results["is_ai"] = combined_prob > 0.5

    return results


def detect_from_input(input_str: str) -> dict:
    """自动判断输入类型并检测"""
    input_str = input_str.strip()

    if is_url(input_str):
        text = extract_text(input_str)
        if text is None:
            return {
                "is_ai": None,
                "probability": None,
                "source": "url",
                "text": None,
                "error": "无法提取网页内容"
            }
        result = detect(text)
        return {
            **result,
            "source": "url",
            "text": text,
            "error": None
        }
    else:
        result = detect(input_str)
        return {
            **result,
            "source": "text",
            "text": input_str,
            "error": None
        }
