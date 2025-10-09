1) pip install uv mcp
2) create project folder, go into it using vs code into that folder
3) uv init
4) uv add mcp
5) build server in that folder
---- install the node.js ----
---- create database in your PG Admin and update your neccessary credentials
     into the main.py's configuration section
6) uv run fastmcp dev main.py // To test the server
7) uv run fastmcp run main.py // To run the server
8) uv run fastmcp install claude-desktop main.py // install your server to claude desktop
     --- after this get the uv path and change the claude config file
     --- quit the claude desktop
9) use the claude desktop to use your personal expense tracker for:
          adding, udpating, deleting, summarizing and listing your expense
          in most interactive fashion like an assistant of yours.