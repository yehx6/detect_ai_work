import sys
import io

# 修复Windows控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from detector import detect_from_input, WEIGHTS


def main():
    if len(sys.argv) < 2:
        print("用法: python main.py <文本或URL>")
        print("示例:")
        print('  python main.py "这是一段测试文字"')
        print('  python main.py "https://example.com/article"')
        sys.exit(1)

    input_str = sys.argv[1]
    result = detect_from_input(input_str)

    if result["error"]:
        print(f"错误: {result['error']}")
        sys.exit(1)

    print(f"输入类型: {'网址' if result['source'] == 'url' else '文本'}")
    print(f"文本字数: {len(result['text'])} 字")
    if result["source"] == "url":
        text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
        print(f"提取文本: {text_preview}")

    print()
    print(f"{'='*60}")
    print(f"检测结果: {'AI生成' if result['is_ai'] else '人类撰写'}")
    print(f"综合概率: {result['probability']:.2%}")
    print(f"{'='*60}")
    print()
    print("各维度分析 (7个检测维度):")
    print("-" * 60)
    print(f"  1. GPT检测器(large):  {result['gpt_detector']:.2%} (权重{WEIGHTS['gpt_detector']})")
    print(f"  2. 困惑度分析:        {result['perplexity_prob']:.2%} (权重{WEIGHTS['perplexity']}) [PPL={result['perplexity']:.1f}]")
    print(f"  3. ChatGPT中文:       {result['chatgpt_zh']:.2%} (权重{WEIGHTS['chatgpt_zh']})")
    print(f"  4. GPT检测器(base):   {result['roberta_base']:.2%} (权重{WEIGHTS['roberta_base']})")
    print(f"  5. 学术文本检测:      {result['academic']:.2%} (权重{WEIGHTS['academic']})")
    print(f"  6. 突发性分析:        {result['burstiness_prob']:.2%} (权重{WEIGHTS['burstiness']}) [CV={result['burstiness']:.2f}]")
    print(f"  7. 词汇丰富度:        {result['vocab_richness_prob']:.2%} (权重{WEIGHTS['vocab_richness']}) [TTR={result['vocab_richness']:.2f}]")


if __name__ == "__main__":
    main()
