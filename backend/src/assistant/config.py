import os

from openai import OpenAI

from .retriever import create_retriever
from .transformer import create_transformer

transformer = create_transformer()
retriever = create_retriever()

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
