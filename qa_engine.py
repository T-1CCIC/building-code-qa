import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from collections import Counter
import streamlit as st

@st.cache_resource
def _load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory="./chroma_db_zh", embedding_function=embeddings)
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    load_dotenv("touch.env")
    api_key = os.getenv("ZHIPU_API_KEY")
    client = ZhipuAI(api_key=api_key)
    return vectorstore, reranker, client

vectorstore, reranker, client = _load_resources()  # 缓存后的全局变量

# 扩展查询
def expand_query(query):
    prompt = f"请将以下问题改写成一段规范文档中可能会出现的详细描述，用于检索：\n{query}"
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        expanded = response.choices[0].message.content
        print(f"原问题：{query}\n扩展后:{expanded}")
        return expanded
    except Exception as e:
        print(f"扩展失败，使用原问题。错误：{e}")
        return query

#     
def generate_answer(query, retrieved_docs,history = None):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 构造历史部分
    history_text = ""
    if history:
        # 只取最近几轮，避免 token 过长
        recent_history = history[-5:]  # 最近5轮（可根据需要调整）
        history_text = "历史对话：\n"
        for msg in recent_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                history_text += f"用户：{content}\n"
            elif role == "assistant":
                history_text += f"助手：{content}\n"
        history_text += "\n"

    prompt = f"""{history_text}请根据以下文档内容回答用户的问题。如果文档中没有相关信息，请如实说明。

文档内容：
{context}

问题：{query}
回答："""
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API调用失败:{e}"

# 核心问答函数
def answer_question(query, use_multi_source=False, lambda_mult=0.8,history = None):
    # ---------- 查询扩展 ----------
    expanded_query = expand_query(query)  # 调用扩展函数
    queries = [query, expanded_query]     # 原问题 + 扩展问题
    
    all_docs = []
    for q in queries:
        # 分别用相似度检索，k=15 可调整
        docs = vectorstore.similarity_search(q, k=15)
        all_docs.extend(docs)
    
    # 按文档内容去重（保留第一个出现的）
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    candidate_docs = list(unique_docs)

    # MMR检索
    # candidate_docs = vectorstore.max_marginal_relevance_search(
    #     query, k=30, fetch_k=100, lambda_mult=lambda_mult
    # )   
    # 普通检索                                                         
    # candidate_docs = vectorstore.similarity_search(query,k=30)   
    
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)

    adjusted = []
    for doc, score in scored_docs:
        year = doc.metadata.get('year', 0)
        weight = 1 + (year - 2000) * 0.05
        adjusted.append((doc, score * weight))
    adjusted.sort(key=lambda x: x[1], reverse=True)

    if use_multi_source:
        source_to_docs = {}
        for doc, weighted_score in adjusted:
            src = doc.metadata['source']
            if src not in source_to_docs or weighted_score > source_to_docs[src][1]:
                source_to_docs[src] = (doc, weighted_score)
        multi_source_docs = [doc for doc, _ in sorted(source_to_docs.values(), key=lambda x: x[1], reverse=True)]
        top_docs = multi_source_docs[:3]
    else:
        top_docs = [doc for doc, _ in adjusted[:3]]

    answer = generate_answer(query, top_docs,history)
    return answer, top_docs


