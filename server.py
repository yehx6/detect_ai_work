"""
AI文本检测API服务
启动命令: python server.py
API地址: http://localhost:8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
import re
import uvicorn

from url_extractor import is_url, extract_text

# ==================== FastAPI应用 ====================
app = FastAPI(
    title="AI文本检测API",
    description="多维度检测文本是否为AI生成",
    version="1.0.0"
)

# ==================== 模型配置 ====================
MODELS_CONFIG = {
    "gpt_detector": "openai-community/roberta-large-openai-detector",
    "perplexity": "gpt2",
    "chatgpt_zh": "Hello-SimpleAI/chatgpt-detector-roberta-chinese",
    "roberta_base": "openai-community/roberta-base-openai-detector",
    "academic": "andreas122001/roberta-academic-detector",
}

# ==================== 权重配置 ====================
WEIGHTS = {
    "gpt_detector": 0.20,
    "perplexity": 0.15,
    "chatgpt_zh": 0.20,
    "roberta_base": 0.15,
    "academic": 0.10,
    "burstiness": 0.10,
    "vocab_richness": 0.10,
}

# ==================== 全局模型变量 ====================
models = {}
tokenizers = {}


def load_models():
    """加载所有模型"""
    global models, tokenizers

    print("正在加载模型...")

    # 1. GPT检测器
    print("  - 加载 GPT检测器 (large)...")
    tokenizers["gpt_detector"] = AutoTokenizer.from_pretrained(MODELS_CONFIG["gpt_detector"])
    models["gpt_detector"] = AutoModelForSequenceClassification.from_pretrained(MODELS_CONFIG["gpt_detector"])

    # 2. GPT-2 困惑度模型
    print("  - 加载 GPT-2 困惑度模型...")
    tokenizers["perplexity"] = GPT2Tokenizer.from_pretrained(MODELS_CONFIG["perplexity"])
    models["perplexity"] = GPT2LMHeadModel.from_pretrained(MODELS_CONFIG["perplexity"])
    models["perplexity"].eval()

    # 3. ChatGPT中文检测器
    print("  - 加载 ChatGPT中文检测器...")
    tokenizers["chatgpt_zh"] = AutoTokenizer.from_pretrained(MODELS_CONFIG["chatgpt_zh"])
    models["chatgpt_zh"] = AutoModelForSequenceClassification.from_pretrained(MODELS_CONFIG["chatgpt_zh"])

    # 4. RoBERTa base检测器
    print("  - 加载 GPT检测器 (base)...")
    tokenizers["roberta_base"] = AutoTokenizer.from_pretrained(MODELS_CONFIG["roberta_base"])
    models["roberta_base"] = AutoModelForSequenceClassification.from_pretrained(MODELS_CONFIG["roberta_base"])

    # 5. 学术文本检测器
    print("  - 加载 学术文本检测器...")
    tokenizers["academic"] = AutoTokenizer.from_pretrained(MODELS_CONFIG["academic"])
    models["academic"] = AutoModelForSequenceClassification.from_pretrained(MODELS_CONFIG["academic"])

    print("所有模型加载完成！")


# ==================== 检测函数 ====================

def detect_with_classifier(text: str, model_key: str) -> float:
    """使用分类器检测，返回AI生成概率"""
    inputs = tokenizers[model_key](text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = models[model_key](**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        if probs.shape[1] == 2:
            ai_prob = probs[0][1].item()
        else:
            ai_prob = probs[0][0].item()
    return ai_prob


def calculate_perplexity(text: str) -> float:
    """使用GPT-2计算文本困惑度"""
    encodings = tokenizers["perplexity"](text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = models["perplexity"](**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())
    return perplexity


def perplexity_to_probability(perplexity: float) -> float:
    """将困惑度转换为AI生成概率"""
    center = 45
    scale = 0.1
    prob = 1 / (1 + math.exp(scale * (perplexity - center)))
    return prob


def calculate_burstiness(text: str) -> float:
    """计算文本突发性（句子长度方差）"""
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
    """计算词汇丰富度 (Type-Token Ratio)"""
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text)
    if len(tokens) < 10:
        return 0.5
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens)
    return ttr


def vocab_richness_to_probability(ttr: float) -> float:
    """将词汇丰富度转换为AI生成概率"""
    center = 0.55
    scale = 8
    prob = 1 / (1 + math.exp(scale * (ttr - center)))
    return prob


def detect(text: str) -> dict:
    """多维度检测文本是否为AI生成"""
    results = {}

    # 1. GPT通用检测器
    results["gpt_detector"] = detect_with_classifier(text, "gpt_detector")

    # 2. 困惑度分析
    perplexity = calculate_perplexity(text)
    results["perplexity"] = perplexity
    results["perplexity_prob"] = perplexity_to_probability(perplexity)

    # 3. ChatGPT中文检测器
    results["chatgpt_zh"] = detect_with_classifier(text, "chatgpt_zh")

    # 4. RoBERTa base检测器
    results["roberta_base"] = detect_with_classifier(text, "roberta_base")

    # 5. 学术文本检测器
    results["academic"] = detect_with_classifier(text, "academic")

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
    results["text_length"] = len(text)

    return results


# ==================== API请求/响应模型 ====================

class DetectRequest(BaseModel):
    text: str = None
    url: str = None


class DetectResponse(BaseModel):
    is_ai: bool
    probability: float
    text_length: int
    source: str
    text_preview: str
    details: dict


# ==================== API路由 ====================

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_models()


@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "AI文本检测API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - 检测文本或URL",
            "/health": "GET - 健康检查"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "models_loaded": len(models) > 0}


@app.post("/detect", response_model=DetectResponse)
async def detect_api(request: DetectRequest):
    """检测文本是否为AI生成"""

    # 确定输入类型和文本
    if request.url:
        text = extract_text(request.url)
        if text is None:
            raise HTTPException(status_code=400, detail="无法提取网页内容")
        source = "url"
    elif request.text:
        text = request.text
        source = "text"
    else:
        raise HTTPException(status_code=400, detail="请提供text或url参数")

    # 执行检测
    result = detect(text)

    # 构造响应
    text_preview = text[:100] + "..." if len(text) > 100 else text

    return DetectResponse(
        is_ai=result["is_ai"],
        probability=result["probability"],
        text_length=result["text_length"],
        source=source,
        text_preview=text_preview,
        details={
            "gpt_detector": result["gpt_detector"],
            "perplexity": result["perplexity"],
            "perplexity_prob": result["perplexity_prob"],
            "chatgpt_zh": result["chatgpt_zh"],
            "roberta_base": result["roberta_base"],
            "academic": result["academic"],
            "burstiness": result["burstiness"],
            "burstiness_prob": result["burstiness_prob"],
            "vocab_richness": result["vocab_richness"],
            "vocab_richness_prob": result["vocab_richness_prob"],
        }
    )


@app.get("/weights")
async def get_weights():
    """获取当前权重配置"""
    return WEIGHTS


# ==================== 启动服务 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("AI文本检测API服务")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
