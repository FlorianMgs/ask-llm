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
