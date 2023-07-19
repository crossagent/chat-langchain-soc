from langchain.agents import tool
from langchain.tools import BaseTool
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources.loading import (
    load_qa_with_sources_chain,
    BaseCombineDocumentsChain,
)
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List

def ingest():
        FILE_PATH = "C:/fandom_scraper/data"

        import os

        all_docs = []

        # 获取 `datapath` 目录下的所有文件
        files = os.listdir(FILE_PATH)

        header_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header_to_split_on)

        chunk_size = 700
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

        # 遍历所有文件
        for file in files:

            # 打开文件
            with open(os.path.join(FILE_PATH, file), "r", encoding="utf-8") as f:

                # 读取文件内容
                content = f.read()

                docs = markdown_splitter.split_text(content)

                # 寻找文件名中的'('和')'的索引
                start_index = file.find("(")
                end_index = file.find(")")

                # 获取文件名和类别
                name = file[:start_index]
                categories = file[start_index+1:end_index].split(",")

                for doc in docs:
                    doc.metadata["source"] = name
                    doc.metadata["categories"] = categories

                split_docs = text_splitter.split_documents(docs)

                all_docs += split_docs

        # TODO：使用向量数据库来预先分拣最相似的片段
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        from langchain.vectorstores import FAISS
        db = FAISS.from_documents(all_docs, embeddings)

        db.save_local("faiss_index")

#暂时使用软关联
def getToolCategory():
    pass

class BaseRetrievesTool(BaseTool):
    
    qa_chain: BaseCombineDocumentsChain
    summary_chain: BaseCombineDocumentsChain

    def _run(self, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""

        # TODO：使用向量数据库来预先分拣最相似的片段
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        from langchain.vectorstores import FAISS
        db = FAISS.load_local("faiss_index", embeddings)

        docs = db.similarity_search_with_score(question, k=4)

        # 获取每个 doc 的 page_content 并连接在一起
        content_text = ""

        metadata_text = ""
        i = 0

        for doc, score in docs:
            if score < 0.55:
                i += 1            
                content_text += f"{i}.{doc.page_content}"
                source  = ""            
                section = ""
                category = ""
                for key, value in doc.metadata.items():
                    if key == "source":
                        source = value
                    elif key == "categories":
                        category = ','.join(value)
                    else:
                        section += value + "-"

                section = section.rstrip("-")  # 移除最后一个连接符'-'

                # 按照指定格式，将 metadata 组织在一起
                metadata = f"{i}.{source} 文件中的 {section}段落，相关度{score}，分类{category}" + "\n"

                metadata_text += metadata

        results_docs = [
            Document(page_content=content_text, metadata={"source": metadata_text})
        ]

        print(results_docs)

        return self.summary_chain(
            {"input_documents": results_docs, "question": question},
            return_only_outputs=True,
        )

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError    
    

class BuildingTool(BaseRetrievesTool):
    name = "buildingTool"
    description = (
        "information about building,include cagerate about Gameplay, Mechanics, Building, and EXPLOSIVE AMMO"
    )

class EnvironmentTool(BaseRetrievesTool):
    name = "EnvironmentTool"
    description = (
        "information about building,include cagerate about Gameplay, Mechanics, Building, and EXPLOSIVE AMMO"
    )

class MechanicsTool(BaseRetrievesTool):
    name = "MechanicsTool"
    description = (
        "information about building,include cagerate about Gameplay, Mechanics, Building, and EXPLOSIVE AMMO"
    )

class SystemTool(BaseRetrievesTool):
    name = "SystemTool"
    description = (
        "information about building,include cagerate about Gameplay, Mechanics, Building, and EXPLOSIVE AMMO"
    )    

if __name__ == "__main__":
    #ingest()
    from langchain.chat_models import ChatOpenAI 
    llm_lookup = ChatOpenAI(temperature=0, model="gpt-4")

    qa_chain= load_qa_with_sources_chain(llm_lookup)
    summary_chain = load_qa_with_sources_chain(llm_lookup)

    buildTool = BuildingTool(qa_chain = qa_chain, summary_chain = summary_chain)

    input = {"question": "我第一天干什么"}
    observation = buildTool.run(input)
    print(observation)

