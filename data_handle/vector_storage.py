import os
from langchain_community.embeddings.modelscope_hub import ModelScopeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from data_handle import data_process

"""
完成文档的嵌入
TODO: 
- 暴露vector_store函数给上层，方便增量更新
- 兼容更多的文件格式，目前仅支持markdown、pdf
"""


def vector_store():
    # 将 folder_path 下所有训练集储存在file_paths里
    file_paths = []
    folder_path = '../doc'
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # 调用处理函数完成数据清洗和数据分割
    split_docs = []
    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            split_docs.extend(data_process.handle_pdf(file_path))
        elif file_type == 'md':
            split_docs.extend(data_process.handle_md(file_path))

    vectordb = get_vector_db()
    vectordb.add_documents(split_docs)
    return vectordb


def get_vector_db():
    persist_directory = './vector_db/chroma'
    vectordb = Chroma(
        embedding_function=ModelScopeEmbeddings(model_id="thomas/text2vec-large-chinese"),
        persist_directory=persist_directory
    )
    return vectordb


if __name__ == '__main__':
    question = "什么是南瓜书？"
    mmr_docs = get_vector_db().max_marginal_relevance_search(question, k=3)

    print(f"检索到的内容数：{len(mmr_docs)}")
    for i, sim_doc in enumerate(mmr_docs):
        print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content}", end="\n--------------\n")
