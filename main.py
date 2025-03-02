from ollamahelpers.manager import OllamaServerCtx
from ollamahelpers import defaults
import ollama
import typer


app = typer.Typer(add_completion=False)

@app.command()
def main(
    prompt:str, 
    host:str=defaults.REMOTEHOST
):
    try:
        with OllamaServerCtx(host=host):
            client = ollama.Client(host)
            response = client.chat(
                model='llama3.2',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    },
                ],
                # format='json'
                format= {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string"
                        },
                        "answer": {
                            "type": "string"
                        },
                    },
                    "required": [ "question", "answer", ] 
                }
            )
            print(response.message.content)
    except TimeoutError as tme:
        print(f'FATAL| {tme}')


if __name__ == "__main__":
    app()