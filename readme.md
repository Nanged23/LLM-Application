# 项目简介

本项目是大模型应用开发的 demo 级实践案例 (๑•͈ᴗ•͈)

🥳 在线体验地址: https://nanged.streamlit.app/

## 1. 目录结构

```markdown
.
├── __pycache__
├── data_handle
│ ├── data_process.py
│ └── vector_storage.py
├── doc
│ ├── 1. 简介 Introduction.md
│ ├── 2. 提示原则 Guidelines.md
│ ├── 3. 迭代优化 Iterative.md
│ ├── 4. 西瓜书简介.md
│ └── pumpkin_book.pdf
├── rag_model
│ └── qa_chain.py
├── raw_model
│ └── tongyi_llm.py
├── readme.md
├── requirements.txt
├── streamlit_app.py
├── vector_db
│ └── chroma
│ ├── bd6cccf7-8630-4892-b7ea-171b5afa9e88
│ │ ├── data_level0.bin
│ │ ├── header.bin
│ │ ├── length.bin
│ │ └── link_lists.bin
│ └── chroma.sqlite3
└── 流程图.png

```

## 2. 相关介绍

- 大模型问答应用的整体流程👉 流程图.png
- 启蒙教程👉 https://datawhalechina.github.io/llm-universe/#/

## 3. 整体环境

- Python3.9
- 安装依赖

```shell
pip install -r requirements.txt
```

## 4. 项目内具体细节

- 整体使用 Langchain 框架完成链路实现，对于部署，使用 streamlit 包完成；
- 对于 Embedding 部分，使用 ModelScope开源 Embedding
  模型。[官方文档](https://help.aliyun.com/document_detail/2668336.html)
- 对于大语言模型(LLM)的选择，使用 qwen-turbo
  模型。[官方文档](https://help.aliyun.com/zh/dashscope/developer-reference/api-details)
- 如果使用部署的方式打开，页面中如果不输入key,则默认采用<kbd>.env</kbd>文件中的key；
- 由于是demo项目，对于算法中涉及的参数，没有严格取值；
