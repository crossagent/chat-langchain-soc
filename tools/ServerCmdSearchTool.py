import json
import nest_asyncio
import asyncio
from langchain.agents import tool
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI

# Needed synce jupyter runs an async eventloop
nest_asyncio.apply()

async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            # 找到目标table节点
            markdown_node = soup.find('div', {'class': 'markdown', 'id': 'pagecontent'})
            table = markdown_node.find('table')

            # 获取表头行
            header_row = table.find('tr')

            # 获取所有数据行
            data_rows = table.find_all('tr')[1:]  # 假设第一行是表头行

            # 遍历每个字段，并添加表头和冒号
            for row in data_rows:
                cells = row.find_all('td')
                for i, cell in enumerate(cells):
                    header_text = header_row.find_all('th')[i].get_text()
                    text = cell.get_text().strip()
                    cell.string = f"{header_text}: {text}"

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results


def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)


@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))


from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import Field
from langchain.chains.qa_with_sources.loading import (
    load_qa_with_sources_chain,
    BaseCombineDocumentsChain,
)


def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=2000,
        chunk_overlap=20,
        length_function=len,
    )

from chains.soc_question_generate import QuestionGenerateChain

class SeverGmCmdTool(BaseTool):
    name = "search_severCommand"
    description = (
        "find admin command in the rust game sever"
    )
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=_get_text_splitter
    )
    qa_chain: BaseCombineDocumentsChain
    summary_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        url = "https://wiki.facepunch.com/rust/useful_commands"
        #url = "https://www.corrosionhour.com/rust-admin-commands/"
        
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []

        # TODO: Handle this with a MapReduceChain
        # for i in range(0, len(web_docs), 4):
        #     input_docs = web_docs[i : i + 4]
        #     window_result = self.qa_chain(
        #         {"input_documents": input_docs, "question": question},
        #         return_only_outputs=True,
        #     )
        #     print(window_result)
        #     results.append(f"Response from window {i} - {window_result}")

        # TODO：使用向量数据库来预先分拣最相似的片段
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        from langchain.vectorstores import FAISS
        db = FAISS.from_documents(web_docs, embeddings)

        docs = db.similarity_search_with_score(question, k=4)

        print(docs)

        results = [doc.page_content for doc, score in docs]

        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
        return self.summary_chain(
            {"input_documents": results_docs, "question": question},
            return_only_outputs=True,
        )

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError
    

if __name__ == "__main__":
    import asyncio

    llm_lookup = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose= True)
    #llm_lookup = ChatOpenAI(temperature=0, model="gpt-4")

    from prompts.search_prompt import PROMPT as SEARCH_PROMPT
    qa_config = {"prompt" : SEARCH_PROMPT}
    #qa_config = {}
    qa_chain= load_qa_with_sources_chain(llm_lookup, **qa_config)

    from prompts.summary_prompt import PROMPT as SUMMARY_PROMPT
    summary_config = {"prompt" : SUMMARY_PROMPT}
    summary_chain = load_qa_with_sources_chain(llm_lookup, **summary_config)

    query_rust_tool = SeverGmCmdTool(qa_chain = qa_chain, summary_chain = summary_chain, verbose = True)

    input = json.loads('{ "url": "https://rust.facepunch.com/", "question": "如何查询我的坐标" }')
    observation = query_rust_tool.run(input)

    print(observation)