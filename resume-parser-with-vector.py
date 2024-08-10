from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

class ResumeInterviewer:
    vector_store = None
    chain = None
    #text_splitter=None
    def _init_(self):
        self.model = Ollama(model="llama3.1:8b", base_url="http://localhost:11434")

        self.prompt = PromptTemplate.from_template(
            """
            You are a helpful HR assistant that analyzes resumes from different candidates.
            Based on the following resume content, generate 5 relevant interview questions.
            Make sure the questions are diverse and cover different aspects of the candidate's experience and skills.

            Resume Content: {context}
            Generate 5 Interview Questions:
            """
        )

    def ingest(self, text_file_path: str):
        loader = TextLoader(text_file_path, encoding='utf-8')
        data = loader.load()
        chunks = text_splitter.split_documents(data)
        chunks = filter_complex_metadata(chunks)

        embeddings = OllamaEmbeddings(model="llama3.1:8b", base_url="http://localhost:11434")

        self.vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

        self.chain = (
                {"context": lambda _: self.vector_store.similarity_search("", k=3), "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
        )

    def generate_questions(self):
        if not self.chain:
            return "Please add a text document first."
        return self.chain.invoke("Generate 5 interview questions based on this resume.")

    def clear(self):
        self.vector_store = None
        self.chain = None


# Main execution
if __name__ == "__main__":
    interviewer = ResumeInterviewer()
    resume_path = "resume.txt"

    print("Ingesting resume...")
    interviewer.ingest(resume_path)

    print("Generating interview questions...")
    questions = interviewer.generate_questions()

    print("Interview Questions:")
    print(questions)