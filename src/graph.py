from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float

def build_rag_graph(index_name: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

    # Define Nodes
    def retrieve_node(state: AgentState):
        docs = retriever.invoke(state["question"])
        context_texts = [d.page_content for d in docs]
        return {"context": context_texts}

    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
If the context does not contain enough info, state 'I cannot answer based on the provided document.'

Context:
{context_str}

Question: {state['question']}"""

        response = llm.invoke(prompt)

        # Simple score heuristic based on retrieved context availability
        confidence = 0.95 if len(state["context"]) > 0 else 0.0

        return {"answer": response.content, "score": confidence}

    # Build Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
