from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

"""
完成文档的清洗和分割
"""


def handle_pdf(file_path):
    """
    处理 pdf 文档
    :param file_path: 文件所在路径
    :return: 清洗后的文档
    """
    loader = PyMuPDFLoader(file_path)
    pdf_pages = loader.load()
    print(f"pdf 文件一共包含 {len(pdf_pages)} 页")

    # 数据清洗：清除\n 点符 空格
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    for pdf_page in pdf_pages:
        pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
        pdf_page.page_content = pdf_page.page_content.replace('•', '').replace(' ', '')
    return split_docs(pdf_pages)


def handle_md(file_path):
    """
    处理 md 文档
    :param file_path: 文件所在路径
    :return: 清洗后的文档
    """
    loader = UnstructuredMarkdownLoader(file_path)
    md_pages = loader.load()
    print(f"md 文件一共包含 {len(md_pages)} 页")
    # 数据清洗
    for md_page in md_pages:
        md_page.page_content = md_page.page_content.replace('\n\n', '\n')
    return split_docs(md_pages)


def split_docs(docs):
    """
    完成文本分割
    :param docs: 清洗后的文档
    :return: 分割后的文档 <list>
    """
    CHUNK_SIZE = 500  # 知识库中单段文本长度
    OVERLAP_SIZE = 50  # 知识库中相邻文本重合长度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
    print(type(split_docs))
    return split_docs
