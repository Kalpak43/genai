from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from typing import Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env file into environment
api_key = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")


class StepResponse(BaseModel):
    step: Literal["START", "PLAN", "OUTPUT"] = Field(
        ..., description="The current stage."
    )
    content: str = Field(
        ..., description="The string for this step."
    )


class StepResponseList(BaseModel):
    steps: list[StepResponse] = Field(
        ..., description="A list of step responses"
    )


parser = JsonOutputParser(pydantic_object=StepResponseList)

SYSTEM_PROMPT = """
    You're an expert AI Assistant in resolving user queries using chain of thought
    You work on START, PLAN and OUPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think enough PLAN has been done, finally you can give an OUTPUT.

    Rules:
     - Strictly follow the given JSON format.
     - Only run one step at a time.
     - The sequence of steps is START(where user gives an input), PLAN(that can be multiple times) and finally OUTPUT(which is going to be displayed to the user).

    Example:
    Q: Hey, Can you solve 2 + 3 * 5 / 10?
    A: {
        "steps": [
            {
                "step": "START",
                "content": "Hey, Can you solve 2 + 3 * 5 / 10?"
            },
            {
                "step": "PLAN",
                "content": "I need to solve the expression 2 + 3 * 5 / 10."
            },
            {
                "step": "PLAN",
                "content": "I will use the order of operations to solve it. I will follow the BODMAS rule."
            },
            {
                "step": "PLAN",
                "content": "Yes, the BODMAS rule is correct here."
            },
            {
                "step": "PLAN",
                "content": "first we multiply 3 and 5 which is 15"
            },
            {
                "step": "PLAN",
                "content": "Now the new equation is 2 + 15 / 10"
            },
            {
                "step": "PLAN",
                "content": "We must perform the division first, so we divide 15 by 10 which is 1.5"
            },
            {
                "step": "PLAN",
                "content": "Now the new equation is 2 + 1.5"
            },
            {
                "step": "PLAN",
                "content": "Finally we add 2 and 1.5 which is 3.5"
            },
            {
                "step": "PLAN",
                "content": "Great, I have the answer. The answer is 3.5"
            },
            {
                "step": "OUTPUT",
                "content": "3.5"
            }
        ]
    }
    
"""

prompt = PromptTemplate(
    template="""
    {system_prompt}

    {format_instructions}
    
    User Input:
    {query}
    """,
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "system_prompt": SYSTEM_PROMPT
    }
)

chain = prompt | model | parser

current_input = "How to implement parallel progressive image loading server using nodejs"

response = chain.invoke({"query": current_input})
print(response)
