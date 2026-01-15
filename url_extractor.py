from urllib.parse import urlparse
import requests
import trafilatura


def is_url(text: str) -> bool:
    """判断输入是否为URL"""
    try:
        result = urlparse(text.strip())
        return result.scheme in ('http', 'https') and bool(result.netloc)
    except Exception:
        return False


def _extract_with_requests(url: str) -> str | None:
    """使用requests获取网页内容"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        text = trafilatura.extract(response.text)
        return text
    except Exception:
        return None


def _extract_with_browser(url: str) -> str | None:
    """使用无头浏览器获取动态渲染的网页内容"""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until='networkidle', timeout=30000)

            # 等待页面加载完成
            page.wait_for_timeout(2000)

            # 获取页面HTML
            html = page.content()
            browser.close()

            # 使用trafilatura提取正文
            text = trafilatura.extract(html)
            return text
    except Exception:
        return None


def _is_valid_content(text: str | None) -> bool:
    """检查提取的内容是否有效（非空且长度足够）"""
    if text is None:
        return False
    # 过滤掉明显的无效内容（如只有UI元素）
    invalid_keywords = ['微信扫一扫', '使用小程序', '环境异常', '去验证', '轻触查看']
    clean_text = text
    for keyword in invalid_keywords:
        clean_text = clean_text.replace(keyword, '')
    # 有效内容应该至少有50个字符
    return len(clean_text.strip()) >= 50


def extract_text(url: str) -> str | None:
    """
    从URL提取正文内容
    优先使用requests，如果内容无效则使用无头浏览器
    返回None表示提取失败
    """
    # 首先尝试使用requests
    text = _extract_with_requests(url)

    if _is_valid_content(text):
        return _clean_text(text)

    # 如果requests获取的内容无效，尝试使用无头浏览器
    text = _extract_with_browser(url)

    if text:
        return _clean_text(text)

    return None


def _clean_text(text: str) -> str:
    """清理提取的文本，移除UI元素等无关内容"""
    # 需要完全移除的UI文本片段
    ui_remove_patterns = [
        '轻触查看原文',
        '向上滑动看下一个',
        '知道了',
        '微信扫一扫',
        '使用小程序',
        '使用完整服务',
        '轻点两下取消赞',
        '轻点两下取消在看',
        '微信扫一扫可打开此内容',
        '分析',
    ]

    # 先移除这些片段
    for pattern in ui_remove_patterns:
        text = text.replace(pattern, '')

    # 需要整行跳过的UI文本
    ui_line_patterns = [
        '取消', '允许', '分享', '收藏', '在看', '赞',
        '视频', '小程序', '×', '留言', '听过',
        '可打开此内容，',
    ]

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # 跳过空行
        if not line:
            continue
        # 跳过纯UI文本行
        if line in ui_line_patterns:
            continue
        # 跳过很短的无意义行
        if len(line) <= 2 and line in ['：', '，', '。', '、', '，', '。']:
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)
