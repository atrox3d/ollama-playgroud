import ollama
import typer
import pydantic
import jsonschema


from ollamahelpers.manager import OllamaServerCtx
from ollamahelpers import defaults

app = typer.Typer(add_completion=False)


class Answer(pydantic.BaseModel):
    question    : str = pydantic.Field(description='the exact original user question')
    answer      : str = pydantic.Field(description='the assistant answer to the question')



@app.command()
def main(
    prompt:str,
    remote:bool=False,
    host:str=None,
    stop:bool=True
):
    try:

        server = defaults.HOST if not remote else host or defaults.REMOTEHOST
        stop = stop if not remote else False
        
        print(f'server: {server}')
        
        with OllamaServerCtx(
                    host=server,
                    stop=stop
            ):
            client = ollama.Client(server)
            response = client.chat(
                model='llama3.2',
                messages=[
                    {
                        'role': 'system',
                        'content': 'answer with the exact user question and the answer'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    },
                ],
                # format='json'
                # format= {
                #     "type": "object",
                #     "properties": {
                #         "question": {
                #             "type": "string"
                #         },
                #         "answer": {
                #             "type": "string"
                #         },
                #     },
                #     "required": [ "question", "answer", ] 
                # }
                format=Answer.model_json_schema()
            )
            print(response.message.content)
    except TimeoutError as tme:
        print(f'FATAL| {tme}')


if __name__ == "__main__":
    app()