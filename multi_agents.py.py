import sympy as sp
import numpy as np
import asyncio
from typing import Union, List
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
import json
load_dotenv()


class OutputModelDetails(BaseModel):
    equation_to_solve: str
    arguments: str
    method_used: str
    solved_response: Union[int, str]

class OutputModel(BaseModel):
    details: list[OutputModelDetails]


@function_tool
def quadratic_solver(a: float, b: float, c: float):
    """Solve ax^2 + bx + c = 0"""
    x = sp.symbols('x')
    expr = a * x ** 2 + b * x + c
    roots = sp.solve(expr, x)
    return [float(root) for root in roots]


@function_tool
def matrix_multiplier(matrix1: List[List[float]], matrix2: List[List[float]]):
    """Multiply two matrices"""
    result = np.matmul(np.array(matrix1), np.array(matrix2))
    return result.tolist()


@function_tool
def eigen_solver(matrix: List[List[float]]):
    """Compute eigenvalues and eigenvectors of a matrix"""
    eigenvalues, eigenvectors = np.linalg.eig(np.array(matrix))
    return {
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist()
    }


quadratic_agent = Agent(
    name="Quadratic Agent",
    instructions="""
You solve quadratic equations of the form ax² + bx + c = 0.
Use `quadratic_solver` tool and solve the users query
- extract the arguments from user query:
    -a:float
    -b:float
    -c:float


""",
    model="litellm/gemini/gemini-2.0-flash-lite",
    tools=[quadratic_solver],
    output_type=OutputModelDetails
)


matrix_agent = Agent(
    name="Matrix Multiplication Agent",
    instructions="""
You multiply 2D matrices.

Use `matrix_multiplier` with two 2D float list inputs.

For matrix multiplication:
- Extract two matrices from user query.
- Use matrix_agent_wrapper(matrix1, matrix2).
- the tool requires 2 matrices 
- matrices should be list of lists
- matrix1: list of lists
- matrix2: list of lists


""",


    model="litellm/gemini/gemini-2.0-flash-lite",
    tools=[matrix_multiplier],
    output_type=OutputModelDetails
)


eigen_agent = Agent(
    name="Eigen Agent",
    instructions="""
You compute eigenvalues and eigenvectors.

Use `eigen_solver` with a square float matrix.

- you should extract the matrix from the user query and then give it to the tool
- the tool accepts matrix i.e a list of lists
- input_matrix: List[List[float]]
""",
    model="litellm/gemini/gemini-2.0-flash-lite",
    tools=[eigen_solver],
    output_type=OutputModelDetails
)


@function_tool
async def quadratic_agent_wrapper(query):
    response = await Runner.run(quadratic_agent, query)
    return response.final_output


@function_tool
async def matrix_agent_wrapper(query):
    response = await Runner.run(matrix_agent, query)
    return response.final_output


@function_tool
async def eigen_agent_wrapper(query):
    response = await Runner.run(eigen_agent, query)
    return response.final_output


orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions="""
    You are a math orchestrator agent that routes the user query to the correct tool.
    Use all the tools at your disposel to get the correct results.
    The final response should:
    - Be a list of `OutputModelDetails`.


""",
    model="litellm/gemini/gemini-2.0-flash-lite",
    tools=[quadratic_agent_wrapper, matrix_agent_wrapper, eigen_agent_wrapper],
    output_type=OutputModel,
)

async def main():
    while True:
        user_input = input("Query: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        try:
            result = await Runner.run(orchestrator_agent, user_input)
            print(" Final Response:")
            print(result.final_output)
            print(json.dumps(result.final_output.model_dump(), indent=4))
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
