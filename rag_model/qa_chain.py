import os
import certifi
from raw_model.tongyi_llm import DashLLM
from data_handle import vector_storage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

"""
构建RAG问答调用模型
"""
os.environ['SSL_CERT_FILE'] = certifi.where()

template = """
你是一个[答疑助手]，根据用户提出的问题进行回答。用户问题如下：
```
问题: {query}
```
TODO:
1. 首先查询数据库和所知道的知识(knowledgeDB)回答，如果不存在答案，则 is_from_knowledgeDB 参数为否,就请自行回复或编造答案。
2. 回答的内容记作参数 answer ,请保持回答内容使用简体中文，保证清晰易读。
3. 保证你的回答按照下面的回答模板。
```
回答模板：
[
来自知识库：<is_from_knowledgeDB>\n
回答内容：<answer>\n
致谢：感谢您的提问~
]
```
"""


def get_qa_chain(query, api_key):
    dic = {"question": template.format(query=query)}
    qa = ConversationalRetrievalChain.from_llm(
        llm=DashLLM(api_key=api_key),
        retriever=vector_storage.get_vector_db().as_retriever(search_type="mmr"),
        memory=ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True),
        return_source_documents=True)
    return qa.invoke(dic)['answer']


if __name__ == "__main__":
    query1 = "什么是南瓜书"
    result = get_qa_chain(query1, None)
    print("大模型+知识库后回答 question_1 的结果：")
    print(result['answer'])
    question = "这本书的作者是谁？"
    result = get_qa_chain(question, None)
    print("大模型+知识库后回答 question_2 的结果：")
    print(result['answer'])
