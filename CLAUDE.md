# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-generated text detection tool using dual-model weighted approach:
1. **RoBERTa分类器** (权重0.7) - `roberta-large-openai-detector`
2. **困惑度分析** (权重0.3) - GPT-2计算文本困惑度

支持直接输入文本或URL（自动提取网页正文）。

## Dependencies

```bash
pip install -r requirements.txt
```

- `transformers` - HuggingFace models (RoBERTa, GPT-2)
- `torch` - PyTorch
- `trafilatura` - Web content extraction
- `requests` - HTTP requests
- `playwright` - Headless browser for dynamic pages

## Running

```bash
# 检测文本
python main.py "这是一段测试文字"

# 检测网页内容
python main.py "https://example.com/article"
```

## Architecture

```
detector.py       # 双模型检测：分类器 + 困惑度分析
url_extractor.py  # URL识别、网页正文提取（支持无头浏览器）
main.py           # 命令行入口
```

## Detection Method

- **分类器**: RoBERTa-large模型直接输出AI概率
- **困惑度**: GPT-2计算困惑度，低困惑度 → 高AI概率（sigmoid映射）
- **综合概率**: `分类器概率 × 0.7 + 困惑度概率 × 0.3`
