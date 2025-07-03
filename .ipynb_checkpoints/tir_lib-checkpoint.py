from utils import *
import sys
import io
import multiprocessing
import prompts
import importlib
importlib.reload(prompts)
from prompts import *

def add_prompt_at_end(response, hint_value = "Wait, I can use Python to check if my approach is correct and refine it, if necessary.```python\n"):
#     print(HINTS.keys())
    return extract_before_token(extract_before_token(response, "</think>"), "**Final Answer**") + hint_value

def add_output_at_end(response, tool_calls = 0):
    code = extract_python_code(response)
    output_json = run_code_with_timeout(code)
    if len(output_json["error"]) == 0:
        return response + "\n```output\n{}\n```\n".format(output_json["output"].strip("\n")), False if output_json["output"].strip("\n") == "" else True
    else:
        return response + "\n```output\nError: {}\n```\n".format(output_json["error"].strip("\n")) + """Wait, I need to make sure the code is appropriate. Let me correct the code based on the error.```python\n""", False

def run_code_with_timeout(code, timeout=10):
    def execute_code(queue, code):
        """
        Function to execute code and store the output or error in a queue.
        """
        try:
            # Redirect stdout to capture the output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Execute the provided code
                exec(code)
                output = sys.stdout.getvalue()
                queue.put({ "output": output, "error": "" })
            except Exception as e:
                queue.put({ "output": "", "error": str(e) })

            # Restore stdout
            sys.stdout = old_stdout
        except Exception as e:
            queue.put({ "output": "", "error": str(e) })
    
    # Create a queue to capture the result from the subprocess
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code, args=(queue, code))

    # Start the process
    process.start()
    # Wait for the process to finish or timeout
    process.join(timeout)

    if process.is_alive():
        # If the process is still alive after the timeout, terminate it
        process.terminate()
        process.join()
        return { "output": "", "error": "Time limit of 10sec exceeded!" }
    if not queue.empty():
        return queue.get()
    else:
        return { "output": "", "error": "Unknown error" }