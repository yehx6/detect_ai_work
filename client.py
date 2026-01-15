"""
AI文本检测客户端
用法:
  python client.py "文本内容"
  python client.py "https://example.com/article"
"""
import sys
import io
import requests

# 修复Windows控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# API服务地址
API_URL = "http://localhost:8000"

# 权重配置（用于显示）
WEIGHTS = {
    "gpt_detector": 0.20,
    "perplexity": 0.15,
    "chatgpt_zh": 0.20,
    "roberta_base": 0.15,
    "academic": 0.10,
    "burstiness": 0.10,
    "vocab_richness": 0.10,
}


def is_url(text: str) -> bool:
    """判断是否为URL"""
    return text.strip().startswith(('http://', 'https://'))


def detect(input_str: str) -> dict:
    """调用API进行检测"""
    input_str = input_str.strip()

    # 构造请求
    if is_url(input_str):
        data = {"url": input_str}
    else:
        data = {"text": input_str}

    # 调用API
    try:
        response = requests.post(f"{API_URL}/detect", json=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到API服务，请先启动: python server.py"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"API错误: {e.response.text}"}
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}


def print_result(result: dict):
    """格式化打印结果"""
    if "error" in result:
        print(f"错误: {result['error']}")
        return

    print(f"输入类型: {'网址' if result['source'] == 'url' else '文本'}")
    print(f"文本字数: {result['text_length']} 字")
    print(f"提取文本: {result['text_preview']}")

    print()
    print("=" * 60)
    print(f"检测结果: {'AI生成' if result['is_ai'] else '人类撰写'}")
    print(f"综合概率: {result['probability']:.2%}")
    print("=" * 60)
    print()

    details = result["details"]
    print("各维度分析 (7个检测维度):")
    print("-" * 60)
    print(f"  1. GPT检测器(large):  {details['gpt_detector']:.2%} (权重{WEIGHTS['gpt_detector']})")
    print(f"  2. 困惑度分析:        {details['perplexity_prob']:.2%} (权重{WEIGHTS['perplexity']}) [PPL={details['perplexity']:.1f}]")
    print(f"  3. ChatGPT中文:       {details['chatgpt_zh']:.2%} (权重{WEIGHTS['chatgpt_zh']})")
    print(f"  4. GPT检测器(base):   {details['roberta_base']:.2%} (权重{WEIGHTS['roberta_base']})")
    print(f"  5. 学术文本检测:      {details['academic']:.2%} (权重{WEIGHTS['academic']})")
    print(f"  6. 突发性分析:        {details['burstiness_prob']:.2%} (权重{WEIGHTS['burstiness']}) [CV={details['burstiness']:.2f}]")
    print(f"  7. 词汇丰富度:        {details['vocab_richness_prob']:.2%} (权重{WEIGHTS['vocab_richness']}) [TTR={details['vocab_richness']:.2f}]")


def main():
    if len(sys.argv) < 2:
        print("AI文本检测客户端")
        print("=" * 40)
        print("用法: python client.py <文本或URL>")
        print()
        print("示例:")
        print('  python client.py "这是一段测试文字"')
        print('  python client.py "https://example.com/article"')
        print()
        print("注意: 请先启动API服务: python server.py")
        sys.exit(1)

    input_str = sys.argv[1]
    result = detect(input_str)
    print_result(result)


if __name__ == "__main__":
    main()
