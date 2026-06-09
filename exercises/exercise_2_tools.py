"""Bai Tap 2: Them Tools va Knowledge Base.

Hoan thanh cac TODO de them tool va knowledge base entry moi.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm


# Knowledge base
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc", "hop dong"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages; (2) consequential damages; (3) specific performance; "
            "(4) cover damages. Statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao dong", "sa thai", "hop dong lao dong", "labor", "termination"],
        "text": (
            "Theo Bo luat Lao dong Viet Nam 2019, nguoi su dung lao dong co the don phuong "
            "cham dut hop dong trong cac truong hop: (1) nguoi lao dong thuong xuyen khong "
            "hoan thanh cong viec; (2) bi om dau, tai nan da dieu tri 12 thang chua khoi; "
            "(3) thien tai, hoa hoan; (4) nguoi lao dong du tuoi nghi huu."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tim kiem trong knowledge base phap ly."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Khong tim thay thong tin lien quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiem tra thoi hieu khoi kien theo loai vu an.

    Args:
        case_type: Loai vu an (contract, tort, property, labor)
    """
    limits = {
        "contract": "4 nam (UCC § 2-725)",
        "tort": "2-3 nam tuy bang",
        "property": "5 nam",
        "labor": "Tuy loai tranh chap lao dong; can kiem tra theo luat dia phuong.",
    }
    return limits.get(case_type.lower(), "Khong xac dinh thoi hieu cho loai vu an nay.")


async def main():
    load_dotenv()
    llm = get_llm()

    tools = [search_legal_knowledge, check_statute_of_limitations]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    question = "Thoi hieu khoi kien vu vi pham hop dong la bao lau?"

    messages = [
        SystemMessage(
            content=(
                "Ban la chuyen gia phap ly. Su dung tools de tra cuu thong tin. "
                "Neu cau hoi hoi ve thoi hieu, hay goi check_statute_of_limitations."
            )
        ),
        HumanMessage(content=question),
    ]

    print(f"Cau hoi: {question}\n")

    # First LLM call - decide which tools to use
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    # Execute tools if requested
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Goi tool: {tool_call['name']}")
            tool_result = await tool_map[tool_call["name"]].ainvoke(tool_call["args"])
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        # Second LLM call - synthesize final answer
        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\nKet qua:\n{final_response.content}")
    else:
        print(f"\nKet qua:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
