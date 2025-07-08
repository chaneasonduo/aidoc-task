import os
from typing import List, Dict
from docx import Document
from langchain_community.llms.tongyi import Tongyi
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi

# 尝试加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

dashscope_key = os.getenv("DASHSCOPE_API_KEY")
if dashscope_key:
    os.environ["DASHSCOPE_API_KEY"] = dashscope_key
else:
    raise ValueError("DASHSCOPE_API_KEY is not set")

# 1. 文档解析

def extract_text_from_docx(file_path: str) -> List[str]:
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def extract_text_from_md(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def extract_paragraphs(file_path: str) -> List[str]:
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.md'):
        return extract_text_from_md(file_path)
    else:
        raise ValueError('仅支持docx或md文件')

# 2. 基于 LLM 匹配评审要点与文档内容

def llm_match_content(paragraphs: List[str], review_point: str, model_name: str = "qwen-turbo", temperature: float = 0.3, max_tokens: int = 512) -> str:
    """
    让 LLM 作为智能检索器，找出与评审要点最相关的文档内容。
    """
    prompt = f"""
            你是一名文档分析专家。请从下列文档段落中，找出与评审要点最相关的内容。

            评审要点：{review_point}

            文档段落：
            """
    for idx, para in enumerate(paragraphs):
        prompt += f"[{idx+1}] {para}\n"
    prompt += "\n请直接返回最相关的段落原文（如有多个可合并），不要添加解释。"
    messages = [
        SystemMessage(content="你是一名文档分析专家。"),
        HumanMessage(content=prompt)
    ]
    llm = Tongyi(model="qwen-turbo")
    result = llm.invoke(messages)
    return str(result)

# 3. 构造链式思维评审结论

def llm_review_conclusion(review_point: str, matched_content: str, model_name: str = "qwen-turbo", temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    让 LLM 基于评审上下文和要点，输出详细的评审结论。
    """
    prompt = f"""
你是一名文档评审专家。请根据以下评审要点和相关文档内容，给出详细的评审结论，并展示你的推理过程：

评审要点：{review_point}

相关内容：{matched_content}

请分步推理，最后给出结论。
"""
    messages = [
        SystemMessage(content="你是一名文档评审专家。"),
        HumanMessage(content=prompt)
    ]
    llm = Tongyi(model="qwen-turbo")
    result = llm.invoke(messages)
    return str(result)

# 4. 主流程：链式思维文档评审

def review_document_with_chain_of_thought(file_path: str, review_points: List[str], model_name: str = "qwen-turbo") -> Dict[str, Dict[str, str]]:
    paragraphs = extract_paragraphs(file_path)
    review_results = {}
    for point in review_points:
        print(f"开始评审要点：{point}")
        # 第一步：用 LLM 匹配相关内容
        print(f"开始匹配相关内容：{point}")
        matched_content = llm_match_content(paragraphs, point, model_name=model_name)
        print(f"匹配相关内容完成：{point}")
        # 第二步：用 LLM 结合上下文和要点输出评审结论
        print(f"开始输出评审结论：{point}")
        conclusion = llm_review_conclusion(point, matched_content, model_name=model_name)
        print(f"输出评审结论完成：{point}")
        review_results[point] = {
            "matched_content": matched_content,
            "conclusion": conclusion
        }
    return review_results

# 5. main 函数示例
if __name__ == "__main__":
    file_path = "/Users/xiaopang/repo/aidoc-task/resources/test-report/report.md"  # 支持 .docx, .md
    review_points = ["格式规范", "内容完整性", "逻辑性","数据准确性"]
    results = review_document_with_chain_of_thought(file_path, review_points)
    for point, res in results.items():
        print(f"\n==== 评审要点：{point} ====")
        print(f"[链式思维-相关内容]：\n{res['matched_content']}")
        print(f"[链式思维-评审结论]：\n{res['conclusion']}") 