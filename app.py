import streamlit as st
import os
from qa_engine import answer_question, vectorstore, reranker  # 导入必要的函数和资源
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv("touch.env")  # 或 ".env"

api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("未找到 ZHIPU_API_KEY 环境变量，请检查 .env 文件或 Streamlit Secrets 设置。")

st.set_page_config(page_title="建筑规范问答助手", page_icon="🏗️")
st.title("🏗️ 建筑规范智能问答")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
prompt = st.chat_input("请输入你的问题...")
if prompt:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用问答函数
    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成答案..."):
            answer, top_docs = answer_question(prompt, use_multi_source=True, lambda_mult=0.8)
            st.markdown(answer)

            # 显示来源文档块（可折叠）
            with st.expander("📄 查看引用来源"):
                for i, doc in enumerate(top_docs):
                    st.write(f"**块 {i+1}** 来源：`{os.path.basename(doc.metadata['source'])}` 第 {doc.metadata['page']} 页")
                    st.text(doc.page_content[:200] + "...")

    # 保存助手回复到历史
    st.session_state.messages.append({"role": "assistant", "content": answer})