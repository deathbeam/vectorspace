# vectorspace

## Run

```bash
pipx run python-vectorspace
```

## Connect as MCP server

```json
{
  "vectorspace": {
    "args": [
      "run",
      "--spec",
      "python-vectorspace==0.0.6",
      "vectorspace-mcp"
    ],
    "command": "pipx"
  }
}
```

## Development

```bash
uv run vectorspace
```

or

```bash
uv pip install -e .
```

and then

```bash
vectorspace
```

### Formatting

```bash
uvx ruff format
uvx ruff check --fix
```

## API docs

```
http://localhost:8000/docs
```
