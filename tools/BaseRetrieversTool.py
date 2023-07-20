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

from pydantic import BaseModel, Field
from typing import List, Optional, Type
class RustWikiInput(BaseModel):
    question: str = Field()
    category: List[str] = Field()


categories = [
    "Ammunition",
    "Animal",
    "Appliances",
    "Armor",
    "Article management templates",
    "Article stubs",
    "Articles for Deletion",
    "BlogListingPage",
    "Blog posts",
    "Boots",
    "Candidates for deletion",
    "Candidates for speedy deletion",
    "Category templates",
    "Clothes",
    "Collectibles",
    "Community",
    "Construction",
    "Consumables",
    "Copyright",
    "Cosmetic",
    "Damage Types",
    "Deployable",
    "Disambiguations",
    "Door",
    "Environment",
    "Experimental Content",
    "Experimental Icons",
    "Explosives",
    "Files",
    "Forums",
    "Fuel",
    "Gameplay",
    "General wiki templates",
    "Guides",
    "Guides (Legacy)",
    "Guns",
    "Help",
    "Help desk",
    "Hidden categories",
    "Icons",
    "Image needed",
    "Image wiki templates",
    "In-Game Event",
    "In-game Settings",
    "Incendiary",
    "Incomplete Data",
    "Infobox templates",
    "Items",
    "Jacket",
    "Legacy",
    "Legacy Icons",
    "Legacy Items",
    "Legacy Redirects",
    "Light Source",
    "Loot",
    "Maps",
    "Mechanics",
    "Medical",
    "Monuments",
    "Mutated",
    "NPC",
    "New pages",
    "Organization",
    "Pages proposed for deletion",
    "Pages with broken file link",
    "Pages with broken file links",
    "Passive Animal",
    "Policy",
    "Ranged Weapon",
    "Redirect",
    "Removed features",
    "Resource Collection",
    "Rust",
    "Rust Wiki",
    "Seasonal",
    "Shoes",
    "Site administration",
    "Site maintenance",
    "Status Effects",
    "Structure",
    "T-Shirt",
    "Talking",
    "Template documentation",
    "Template icons",
    "Templates",
    "Throwable Weapon",
    "Tools",
    "Trap",
    "Unreleased Content",
    "Videos",
    "Water Container",
    "Weapon Mods",
    "Weapons",
    "Window",
    "XP System"
]

class RustWikiTool(BaseTool):
    name = "RustWiki"
    description = (
        f"useful when you need to search imformation about rust game, the category in args can only be selected from the following type:{','.join(categories)}."
    )
    summary_chain: BaseCombineDocumentsChain

    args_schema:Type[BaseModel] = RustWikiInput

    def _run(self, question: str, category: List[str]) -> str:
        """Useful for browsing websites and scraping the text information."""

        # TODO：使用向量数据库来预先分拣最相似的片段
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        from langchain.vectorstores import FAISS
        db = FAISS.load_local("faiss_index", embeddings)

        final_question = question + f"分类属于{'.'.join(category)}"

        docs = db.similarity_search_with_score(final_question, k=4)

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
    

class BuildingTool(RustWikiTool):
    name = "buildingTool"
    description = (
        "information about building,include cagerate about Gameplay, Mechanics, Building, and EXPLOSIVE AMMO"
    )

class EnvironmentTool(RustWikiTool):
    name = "EnvironmentTool"
    description = (
        "information about building,include cagerate about Gameplay, Mechanics, Building, and EXPLOSIVE AMMO"
    )

class MechanicsTool(RustWikiTool):
    name = "MechanicsTool"
    description = (
        "information about building,include cagerate about Gameplay, Mechanics, Building, and EXPLOSIVE AMMO"
    )

class SystemTool(RustWikiTool):
    name = "SystemTool"
    description = (
        "information about building,include cagerate about Gameplay, Mechanics, Building, and EXPLOSIVE AMMO"
    )    

if __name__ == "__main__":
    ingest()
    from langchain.chat_models import ChatOpenAI 
    llm_lookup = ChatOpenAI(temperature=0, model="gpt-4")

    summary_chain = load_qa_with_sources_chain(llm_lookup)

    buildTool = BuildingTool(summary_chain = summary_chain)

    input = {"question": "我第一天干什么"}
    observation = buildTool.run(input)
    print(observation)

