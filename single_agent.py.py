import os
from openai import OpenAI
from dotenv import load_dotenv 
from pydantic import BaseModel
import numpy as np
import json
import sympy
load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
class IndividualResult(BaseModel):
    expression:str
    arguments:str
    method_used:str
    solved_response:str
class ResultFormat(BaseModel):
    results: list[IndividualResult]
def quadratic_equation(a:float,b:float,c:float)->str:
    try:
        x=sympy.symbols('x')
        equation=sympy.Eq(a*x**2+b*x+c,0)
        solutions=sympy.solve(equation,x)
        solutions_str=[str(sol) for sol in solutions]
        return json.dumps({"roots":solutions_str})
    except Exception as e:
        return json.dumps({"error": f"Failed to solve quadratic equation: {e}"})
def eigen_vectors(matrix_str:str)->str:
    try:
        matrix_list=json.loads(matrix_str)
        matrix=np.array(matrix_list,dtype=float)
        if matrix.shape[0]!=matrix.shape[1]:
            raise ValueError("Matrix must be square for eigenvalue/eigenvector calculation.")
        eigenvalues,eigenvectors=np.linalg.eig(matrix)
        return json.dumps(
            {
                "eigenvalues":eigenvalues.tolist(),
                "eigenvectors": eigenvectors.tolist()
            }
        )
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during eigenvalue/eigenvector calculation: {e}"})
def multiply_matrices(matrix1_str: str, matrix2_str: str) -> str:
    try:
        matrix1_list=json.loads(matrix1_str)
        matrix2_list=json.loads(matrix2_str)
        mat1=np.array(matrix1_list,dtype=float)
        mat2=np.array(matrix2_list,dtype=float)
        if mat1.shape[1]!=mat2.shape[0]:
            raise ValueError("Number of columns in the first matrix must equal the number of rows in the second matrix.")
        result_matrix=np.matmul(mat1,mat2)
        return json.dumps({"result_matrix": result_matrix.tolist()})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during matrix multiplication: {e}"})
tools=[
   {
       "type":"function",
       "function":{
           "name":"quadratic_equation",
           "description":"Solves a quadratic equation of the form ax^2 + bx + c = 0.Provide coefficients a, b, and c.",
           "parameters":{
               "type":"object",
               "properties":{
                   "a":{"type":"number","description": "Coefficient of x^2."},
                   "b": {"type": "number", "description": "Coefficient of x."},
                   "c": {"type": "number", "description": "Constant term."},
               },
               "required":["a","b","c"],
           },
       },
   },
 {
       "type": "function",
       "function": {
           "name": "eigen_vectors",
           "description": "Calculates eigenvalues and eigenvectors for a square matrix. Input must be a JSON string representing a list of lists (e.g., '[[1,2],[3,4]]').",
           "parameters": {
               "type": "object",
               "properties": {
                   "matrix_str": {"type": "string", "description": "The square matrix as a JSON string (e.g., '[[1,2],[3,4]]')."},
               },
               "required": ["matrix_str"],
           },
       },
   },
   {
   "type": "function",
       "function": {
           "name": "multiply_matrices",
           "description": "Multiplies two matrices. Inputs must be JSON strings representing lists of lists (e.g., '[[1,2],[3,4]]').",
           "parameters": {
               "type": "object",
               "properties": {
                   "matrix1_str": {"type": "string", "description": "The first matrix as a JSON string (e.g., '[[1,2],[3,4]]')."},
                   "matrix2_str": {"type": "string", "description": "The second matrix as a JSON string (e.g., '[[5,6],[7,8]]')."},
               },
               "required": ["matrix1_str", "matrix2_str"],
           },
       },
   },
]
available_functions={
    "quadratic_equation": quadratic_equation, 
    "eigen_vectors": eigen_vectors,           
    "multiply_matrices": multiply_matrices,
}
def run_math_conversation(user_prompt:str):
    model="gemini-2.5-flash"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    response=client.chat.completions.create(
       model=model,
       messages=messages,
       tools=tools,
       tool_choice="auto"
    )
    #print(response.choices[0].message)
    response_message=response.choices[0].message
    tool_calls=response_message.tool_calls
    #print(tool_calls)
    if tool_calls:
        print("llm wants to call tool")
        messages.append(response_message)
        print("message---->",messages)
        tool_outputs=[]
        for tool in tool_calls:
            fn_name=tool.function.name
            fn_to_call=available_functions.get(fn_name)
            if fn_to_call:
                try:
                    args=json.loads(tool.function.arguments)
                    tool_output=fn_to_call(**args)
                    print(tool_output)
                    tool_outputs.append(tool_output)
                    print(f"DEBUG: Tool '{fn_name}' returned: {tool_output}")
                    context={
                        "tool_call_id":tool.id,
                        "role":"tool",
                        "name":fn_name,
                        "content":tool_output
                    }
                    print(context)
                    messages.append(context)
                except Exception as e:
                    error_msg = f"Error executing tool '{fn_name}': {e}"
                    context={
                        "tool_call_id":tool.id,
                        "role":"tool",
                        "name":fn_name,
                        "content":json.dumps({"error":error_msg})
                    }
            else:
                error_msg=f"Error: Tool '{fn_name}' not found."
                print(f"DEBUG: {error_msg}")
                messages.append({
                    "tool_call_id":tool.id,
                    "role":"tool",
                    "name":fn_name,
                    "content":json.dumps({"error":error_msg})
                })
        #print("messages------->",messages)
        #print("sending response back to llm")
        second_response=client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=ResultFormat
        )
        #print("second_response--->",second_response.choices)
        #print("###########################")
        #print("hello-->",second_response.choices[0].message.parsed.model_dump_json(indent=2))
            #print("------")
            #print(second_response)
    
        parsed_results=[]
        for choice in second_response.choices:
            #print("choices---->",choice)
            if choice.message.parsed:
                for step in choice.message.parsed.results:
                    parsed_results.append(step.model_dump())
        return parsed_results
    else:
        return json.dumps({"status": "no tool call was made or no final response generated"})

if __name__=="__main__":
    print("Type quit to exit")
    while True:
        user_input=input("Give your prompt >> or type quit to exit")
        if user_input.lower()=="quit":
            print("Bye.....")
            break
        results=run_math_conversation(user_input)
        print("Your answers...")
        for result in results:
            print(json.dumps(result,indent=2))
