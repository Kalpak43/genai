from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env file into environment
api_key = os.getenv("GOOGLE_API_KEY")


model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")


SYSTEM_PROMPT = "You are a helpful maths instructor. You are given a question and you need to answer it. You need to be concise and to the point. Do not answer questions related to other topics."

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="Solve integral of x^2 from 0 to 1"),
]

response = model.invoke(messages)
print(response.content)
