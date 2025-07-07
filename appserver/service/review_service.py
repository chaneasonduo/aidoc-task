import os
from typing import List, Dict
from docx import Document
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi
from langchain_cohere import ChatCohere

# 1. 文档解析
def extract_text_from_docx(file_path: str) -> List[str]:
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def extract_text_from_doc(file_path: str) -> List[str]:
    result = subprocess.run(['antiword', file_path], stdout=subprocess.PIPE)
    text = result.stdout.decode('utf-8')
    return [line.strip() for line in text.splitlines() if line.strip()]

def extract_text_from_md(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def extract_paragraphs(file_path: str) -> List[str]:
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.doc'):
        return extract_text_from_doc(file_path)
    elif file_path.endswith('.md'):
        return extract_text_from_md(file_path)
    else:
        raise ValueError('仅支持docx、doc、md文件')

# 2. embedding模型（通义千问兼容sentence-transformers格式）
# 推荐中文模型：shibing624/text2vec-base-chinese
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'shibing624/text2vec-base-chinese')
embedder = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, normalize_embeddings=True)

# 3. 基于要点与文档内容的embedding相似度匹配
def match_review_points_with_doc(paragraphs: List[str], review_points: List[str], k: int = 3) -> Dict[str, List[str]]:
    para_embeds = get_embeddings(paragraphs)
    if len(para_embeds.shape) == 1:
        para_embeds = para_embeds.reshape(1, -1)
    if para_embeds.shape[0] == 0:
        return {point: ['未找到直接相关内容'] for point in review_points}
    results = {}
    for point in review_points:
        point_embed = get_embeddings([point])
        if len(point_embed.shape) == 1:
            point_embed = point_embed.reshape(1, -1)
        index = faiss.IndexFlatIP(para_embeds.shape[1])
        index.add(para_embeds.astype('float32'))
        D, I = index.search(point_embed.astype('float32'), k)
        top_paras = [paragraphs[i] for idx, i in enumerate(I[0]) if D[0][idx] > 0.2]
        results[point] = top_paras if top_paras else ['未找到直接相关内容']
    return results

# 4. 构造提示词
def build_review_prompt(review_points: List[str], matched_paragraphs: Dict[str, List[str]]) -> str:
    prompt = "你是一名文档评审专家，请根据以下要点和文档内容进行评审并给出结论。\n"
    for point in review_points:
        prompt += f"\n【评审要点】：{point}\n【相关内容】：\n"
        for para in matched_paragraphs[point]:
            prompt += f"- {para}\n"
    prompt += "\n请综合上述内容，逐条给出评审结论。"
    return prompt

# 5. 调用通义千问 LLM（langchain官方社区模型ChatTongyi）
def review_doc_with_embedding(file_path: str, review_points: List[str], model_name: str = "qwen-turbo", temperature: float = 0.7, max_tokens: int = 2000, use_cohere: bool = False, cohere_api_key: str = None, cohere_model: str = "command-r-plus") -> str:
    paragraphs = extract_paragraphs(file_path)
    matched = match_review_points_with_doc(paragraphs, review_points)
    prompt = build_review_prompt(review_points, matched)
    messages = [
        SystemMessage(content="你是一名文档评审专家。"),
        HumanMessage(content=prompt)
    ]
    if use_cohere:
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if api_key:
            llm = ChatCohere(model=cohere_model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
        else:
            llm = ChatCohere(model=cohere_model, temperature=temperature, max_tokens=max_tokens)
    else:
        llm = ChatTongyi(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    result = llm.invoke(messages)
    return str(result)

# 6. main函数示例
if __name__ == "__main__":
    file_path = "your_doc.md"  # 支持 .docx, .doc, .md
    review_points = ["格式规范", "内容完整性", "逻辑性"]
    # 使用 Cohere LLM（需设置 COHERE_API_KEY 环境变量或传入 cohere_api_key）
    # conclusion = review_doc_with_embedding(file_path, review_points, use_cohere=True, cohere_api_key="your_cohere_api_key")
    # 使用通义千问 LLM
    conclusion = review_doc_with_embedding(file_path, review_points)
    print("评审结论：", conclusion) 