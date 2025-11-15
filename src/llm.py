import os
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

class GroqLLM:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not HAS_GROQ:
            self.llm = None
            return
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )

    def generate_response(self, query: str, context: str) -> str:
        if not HAS_GROQ or not self.llm:
            return context[:2000] + ("..." if len(context) > 2000 else "")
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        )
        formatted = prompt_template.format(context=context, question=query)
        messages = [HumanMessage(content=formatted)]
        response = self.llm.invoke(messages)
        return response.content
