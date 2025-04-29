from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.llm_model import OpenSourceLLM

class RAGPipeline:
    def __init__(self, retriever):
        self.llm = OpenSourceLLM().llm
        self.retriever = retriever
        
        self.qa_prompt = PromptTemplate(
            template="""Answer using this context:
            {context}
            Question: {question}
            Answer:""",
            input_variables=["context", "question"]
        )
    
    def answer_question(self, query):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )
        return qa_chain({"query": query})["result"]