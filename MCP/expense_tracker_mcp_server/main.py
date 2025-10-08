from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool
def add_numbers(a,b):
    return a+b

@mcp.tool
def multiply_numbers(a: float,b: float)-> float:
    return a*b

if __name__ == "__main__":
    mcp.run()
