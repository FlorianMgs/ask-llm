## ask-llm: the easiest way to supercharge your apps with LLM!
ask-llm is a very simple yet powerful package that can turn anything into a LLM interaction.  
You just need to decorate a function with `@ask()`, write a prompt into the docstring, give it a return type, and there you go, you got your LLM interaction.  
This takes inspiration from the awesome langchain-decorators package.  
It works out of the box with OpenAI (by setting a `OPENAI_API_KEY` env var), but is compatible with all `BaseChatModel` from Langchain.

## Features
- Write your prompts in docstrings using Jinja templating language. 
- Access your function args in your prompt.  
- (Almost) full access in your prompt to every attributes / properties of objects as soon as it's not a callable.  
- Fully compatible with Langchain / LCEL. Returns either the LLM response, the formatted `ChatPromptTemplate` or a chain `prompt | llm`.  
- Access the decorated function return value inside your prompt. Use `{{ __result__ }}` anywhere in your prompt.  
- Format the LLM answer using Pydantic objects as return type.  
- If passed a Pydantic object as return type, LLM will retry if it fails to answer on the first shot.  
- If using GPT Vision, supports an `image` parameter to send alongside your prompt. You can input your image either as an url, a path, or a base64 string.  
and many more to come...

### Installation
Either using `pip` or by cloning this repo. 
`pip install ask-llm`

Then, import the decorator:
`from ask_llm import ask`

To make it work out of the box with OpenAI, define an env var:
`OPENAI_API_KEY=<your key>`

Now, you are able to use the decorator:
```python
import requests
from functools import cached_property
from ask_llm import ask, BaseAnswer


class BlogArticle(BaseAnswer):
    title: str
    content: str


class WikipediaAPI:
    def __init__(self, title: str):
        self.title = title

    @cached_property
    def wikipedia_article(self) -> str | None:
        try:
            response = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{self.title}"
            )
            return response.json()["extract"]
        except:
            return


class BlogArticleWriter:
    def __init__(
        self,
        title: str,
        keywords: list,
        nb_headings: str,
        nb_paragraphs: str,
        input_bulletpoints: bool,
    ):
        self.title = title
        self.keywords = keywords
        self.nb_headings = nb_headings
        self.nb_paragraphs = nb_paragraphs
        self.input_bulletpoints = input_bulletpoints

        self.wikipedia_api = WikipediaAPI(title=self.title)

    @ask()
    def write_blog_article(self, author: str) -> BlogArticle:
        """
        As an expert copywriter specialized in SEO and content writing, your task is to write a very informative blog article
        about the topic {{ self.title }}.
        You should create {{ self.nb_headings }} highly engaging headings made of {{ self.nb_paragraphs }} paragraphs each.
        Use subheadings and line breaks when appropriate.

        {% if self.input_bulletpoints %}
          You should also include bulletpoints in the article.
        {% endif %}

        {% if self.wikipedia_api.wikipedia_article %}
          Here is a brief summary of the topic:
          {{ self.wikipedia_api.wikipedia_article }}
        {% endif %}

        {% if self.keywords %}
          The following keywords should be included in the article:
          {% for keyword in self.keywords %}
            - {{ keyword }}
          {% endfor %}
        {% endif %}

        The article should be written by {{ author }}.
        """

writer = BlogArticleWriter(
    title="Large language models", 
    keywords=["llm", "open source", "python", "nlp"], 
    nb_headings=3, 
    nb_paragraphs=3, 
    input_bulletpoints=True
)

blog_article = writer.write_blog_article("Florian")
```
Langsmith trace: https://smith.langchain.com/public/0abe7b97-d43e-4c10-9bba-be9f6c2892d6/r


An other example, with an image:
```python
from ask_llm import ask


@ask()
def describe_image(image: str) -> str:
    """
    Describe this image precisely.
    """


description = describe_image(image="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/800px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg")
```
Langsmith trace: https://smith.langchain.com/public/18f39957-93f5-47f2-a264-051c11cca2e8/r

### Settings  
You can pass numerous arguments to the decorator:  
- `call: bool = True` To call the LLM or to return the prepared chain `prompt | llm`  
- `return_prompt_only: bool = False` To return only the formatted `ChatPromptTemplate`  
- `model_name: str = "gpt-4-vision-preview"` The model name  
- `max_tokens: int = 4096` max tokens for the answer  
- `image_quality: str = "high"` If passed an image, the quality parameter for Vision API  
- `chat_model_class: BaseChatModel = ChatOpenAI` The Langchain subclass of `BaseChatModel` to use.  
- `**llm_kwargs` Other kwargs to pass to the `BaseChatModel` if any.  


### Conclusion
Possibilities with this are endless. Hope you're gonna like it!  
More features and examples are coming.  
Special integration with Django is also coming.  
Do not hesitate to iterate and contribute to the project! Please submit PRs and issues 🙏
