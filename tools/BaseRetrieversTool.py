from langchain.agents import tool
from langchain.tools import BaseTool
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources.loading import (
    load_qa_with_sources_chain,
    BaseCombineDocumentsChain,
)
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class BaseRetrievesTool(BaseTool):
    
    qa_chain: BaseCombineDocumentsChain
    summary_chain: BaseCombineDocumentsChain


    def _run(self, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""

        FILE_PATH = "E:/MyProject/fandom_scraper/data"

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

                for doc in docs:
                    doc.metadata["source"] = file

                split_docs = text_splitter.split_documents(docs)

                all_docs += split_docs
        
        results = []

        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(all_docs), 4):
            input_docs = all_docs[i : i + 4]
            window_result = self.qa_chain(
                {"input_documents": input_docs, "question": question},
                return_only_outputs=True,
            )
            print(window_result)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
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

if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI 
    llm_lookup = ChatOpenAI(temperature=0, model="gpt-4")

    qa_chain= load_qa_with_sources_chain(llm_lookup)
    summary_chain = load_qa_with_sources_chain(llm_lookup)

    buildTool = BuildingTool(qa_chain = qa_chain, summary_chain = summary_chain)

    input = {"question": "如何造房子" }
    observation = buildTool.run(input)
    print(observation)

