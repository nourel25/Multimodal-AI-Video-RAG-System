from .BaseController import BaseController
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

class ChunkController(BaseController):
    def __init__(self):
        super().__init__()
        
    def get_file_content(self, file_path: str):
        loader = TextLoader(file_path=file_path, encoding='utf-8')
        return loader.load()
        
    def process_file_content(self, file_content: list, 
                             chunk_size: int=100, overlap_size: int=20):
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
        )
        
        file_content_texts = [
            doc.page_content
            for doc in file_content
        ]
        
        file_content_metadata = [
            doc.metadata
            for doc in file_content
        ]
        
        chunks = text_splitter.create_documents(
            file_content_texts,
            file_content_metadata
        )
        
        return chunks