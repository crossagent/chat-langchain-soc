from langchain.agents import Tool

def fake_func(inp: str) -> str:
    return "foo"


fake_tools = [
    Tool(
        name=f"foo-{i}",
        func=fake_func,
        description=f"a silly function that you can use to get more information about the number {i}",
    )
    for i in range(99)
]
ALL_TOOLS = fake_tools