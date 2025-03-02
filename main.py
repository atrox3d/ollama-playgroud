from ollamahelpers.manager import OllamaServerCtx
from ollamahelpers import defaults
import ollama
import typer


app = typer.Typer(add_completion=False)

@app.command()
def main(host:str=defaults.REMOTEHOST):
    try:
        with OllamaServerCtx(host=host):
            client = ollama.Client(host)
    except TimeoutError as tme:
        print(f'FATAL| {tme}')


if __name__ == "__main__":
    app()