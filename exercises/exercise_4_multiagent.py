"""Bai Tap 4: Them Privacy Agent vao Multi-Agent System.

Hoan thanh bai tap multi-agent voi agent moi chuyen ve privacy/GDPR.
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: gia tri moi ghi de gia tri cu."""
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    final_response: str


def law_agent(state: State) -> dict:
    """Agent phan tich phap ly tong quat."""
    print("[Node: law_agent] Dang phan tich phap ly tong quat...")
    llm = get_llm()
    prompt = f"""Ban la chuyen gia phap ly. Phan tich cau hoi sau:

{state['question']}

Tap trung vao: hop dong, trach nhiem dan su, quyen va nghia vu phap ly.
Tra loi ngan gon duoi 180 tu."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("[Node: law_agent] Done")
    return {"law_analysis": response.content}


def check_routing(state: State) -> list[Send]:
    """Quyet dinh goi agents nao dua tren noi dung cau hoi."""
    question_lower = state["question"].lower()
    tasks: list[Send] = []

    if any(kw in question_lower for kw in ["tax", "irs", "thue", "thuế"]):
        tasks.append(Send("tax_agent", state))

    if any(
        kw in question_lower
        for kw in ["compliance", "sec", "regulation", "regulatory", "tuan thu", "tuân thủ"]
    ):
        tasks.append(Send("compliance_agent", state))

    if any(
        kw in question_lower
        for kw in ["data", "privacy", "gdpr", "ccpa", "du lieu", "dữ liệu", "personal data"]
    ):
        tasks.append(Send("privacy_agent", state))

    if tasks:
        print("[Routing] Goi agents:", ", ".join(task.node for task in tasks))
        return tasks

    print("[Routing] Khong can specialist, chuyen sang aggregate_results")
    return [Send("aggregate_results", state)]


def tax_agent(state: State) -> dict:
    """Agent chuyen ve thue."""
    print("[Node: tax_agent] Dang phan tich thue...")
    llm = get_llm()
    prompt = f"""Ban la chuyen gia thue. Phan tich khia canh thue trong cau hoi:

Cau hoi: {state['question']}
Phan tich phap ly: {state.get('law_analysis', 'N/A')}

Tap trung: IRS, tax evasion, penalties, FBAR, FATCA.
Tra loi ngan gon duoi 160 tu."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("[Node: tax_agent] Done")
    return {"tax_analysis": response.content}


def compliance_agent(state: State) -> dict:
    """Agent chuyen ve compliance."""
    print("[Node: compliance_agent] Dang phan tich tuan thu...")
    llm = get_llm()
    prompt = f"""Ban la chuyen gia compliance. Phan tich khia canh tuan thu:

Cau hoi: {state['question']}
Phan tich phap ly: {state.get('law_analysis', 'N/A')}

Tap trung: SEC, SOX, FCPA, AML, regulatory violations.
Tra loi ngan gon duoi 160 tu."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("[Node: compliance_agent] Done")
    return {"compliance_analysis": response.content}


def privacy_agent(state: State) -> dict:
    """Agent chuyen ve bao ve du lieu ca nhan va GDPR."""
    print("[Node: privacy_agent] Dang phan tich privacy/GDPR...")
    llm = get_llm()
    prompt = f"""Ban la chuyen gia ve GDPR va luat bao ve du lieu ca nhan.

Cau hoi goc: {state['question']}
Phan tich phap ly: {state.get('law_analysis', 'N/A')}

Hay phan tich cac van de ve privacy va GDPR neu co.
Tap trung: personal data, data breach, consent, data subject rights,
notification duties, cross-border transfer, GDPR/CCPA penalties.
Tra loi ngan gon duoi 180 tu."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("[Node: privacy_agent] Done")
    return {"privacy_analysis": response.content}


def aggregate_results(state: State) -> dict:
    """Tong hop ket qua tu tat ca agents."""
    print("[Node: aggregate_results] Dang tong hop ket qua...")
    llm = get_llm()

    sections = []
    if state.get("law_analysis"):
        sections.append(f"PHAN TICH PHAP LY:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"PHAN TICH THUE:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"PHAN TICH TUAN THU:\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"PHAN TICH PRIVACY/GDPR:\n{state['privacy_analysis']}")

    combined = "\n\n---\n\n".join(sections)

    prompt = f"""Tong hop cac phan tich sau thanh mot bao cao phap ly hoan chinh:

{combined}

Cau hoi goc: {state['question']}

Hay tao mot bao cao ngan gon, co cau truc ro rang."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("[Node: aggregate_results] Done")
    return {"final_response": response.content}


def build_graph():
    """Xay dung multi-agent graph."""
    graph = StateGraph(State)

    graph.add_node("law_agent", law_agent)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)

    graph.add_edge(START, "law_agent")
    graph.add_conditional_edges(
        "law_agent",
        check_routing,
        ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"],
    )
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()


async def main():
    load_dotenv()

    question = (
        "Neu cong ty bi ro ri du lieu khach hang, vi pham compliance, "
        "va co van de ve thue, hau qua phap ly la gi?"
    )

    print("=" * 70)
    print("MULTI-AGENT SYSTEM voi Privacy Agent")
    print("=" * 70)
    print(f"\nCau hoi: {question}\n")
    print("Dang xu ly qua cac agents...\n")

    graph = build_graph()

    result = await graph.ainvoke(
        {
            "question": question,
            "law_analysis": "",
            "tax_analysis": "",
            "compliance_analysis": "",
            "privacy_analysis": "",
            "final_response": "",
        }
    )

    print("\n" + "=" * 70)
    print("KET QUA CUOI CUNG")
    print("=" * 70)
    print(result["final_response"])
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
