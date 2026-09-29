import requests
import json

API_URL = "http://localhost:8000/chat"

queries = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?"
]

def run_tests():
    for query in queries:
        print(f"\n[{'-'*50}]")
        print(f"Query: {query}")
        try:
            response = requests.post(API_URL, json={"query": query})
            if response.status_code == 200:
                data = response.json()
                print(f"Final Answer: {data['final_answer']}")
                print(f"Confidence Score: {data['confidence_score']}")
                print(f"Retrieved Context Chunks: {len(data['retrieved_context_chunks'])}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Connection failed: {e}")

if __name__ == "__main__":
    print("Make sure FastAPI is running on http://localhost:8000")
    run_tests()
