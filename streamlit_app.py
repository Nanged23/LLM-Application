import streamlit as st
from raw_model.tongyi_llm import DashLLM
from rag_model import qa_chain


# 使用 streamlit 构建前端页面，官方文档：https://docs.streamlit.io/develop/api-reference/write-magic


def llm_response(input_text, api_key):
    llm = DashLLM(api_key=api_key)
    answer = llm.invoke(input_text)
    return answer


def rag(input_text, api_key):
    return qa_chain.get_qa_chain(input_text, api_key)


if __name__ == "__main__":

    st.title('🦜🔗 LLM 应用示例')
    api_key = st.sidebar.text_input('通义千问api-key', type='password')
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["普通模式", "高级模式"],
        captions=["普通模型，不带历史记录", "特定优化，带历史记录"])
    if selected_method == '普通模式':
        with st.form('my_form'):
            text = st.text_area('🤓今天想提问些什么？', placeholder='为我推荐一道浙江的小吃 🍖')
            submitted = st.form_submit_button('✈️️')
            if not api_key.startswith('sk-'):
                st.toast(
                    '请先在左侧输入以\'sk-\'开头的通义密钥 默认值：sk-0a053959c62840ed94fd1a03324f41fc'
                    '可前往 https://dashscope.console.aliyun.com/apiKey 开通~',
                    icon='⚠')
            if api_key.startswith('sk-') and submitted:
                if text == '':
                    st.toast("提问内容不可为空！")
                answer = llm_response(text, api_key)
                st.info(answer)
    else:  # 调用 RAG 进行回答
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        messages = st.container(height=300)
        if prompt := st.chat_input("Say something"):
            # 将用户输入添加到对话历史中
            st.session_state.messages.append({"role": "user", "text": prompt})
            # 调用 respond 函数获取回答
            answer = rag(prompt, api_key)
            # 检查回答是否为 None
            if answer is not None:
                # 将LLM的回答添加到对话历史中
                st.session_state.messages.append({"role": "assistant", "text": answer})
            # 显示整个对话历史
            for message in st.session_state.messages:
                if message["role"] == "user":
                    messages.chat_message("user").write(message["text"])
                elif message["role"] == "assistant":
                    messages.chat_message("assistant").write(message["text"])

    st.page_link("https://github.com/Nanged23/LLM-Application", label="Github repo", icon="🌎", use_container_width=True)
    st.caption("联系作者：ferdinandlekae46@gmail.com")

# #不带历史记录的问答链
# def get_qa_chain(question:str,api_key:str):
#     vectordb = get_vectordb()
#     llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,api_key = api_key)
#     template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
#         案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
#         {context}
#         问题: {question}
#         """
#     QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
#                                  template=template)
#     qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
#     result = qa_chain({"query": question})
#     return result["result"]
