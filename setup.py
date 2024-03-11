from setuptools import setup, find_packages

setup(
    name="ask-llm",
    version="0.1.12",
    packages=find_packages(),
    url="https://github.com/FlorianMgs/ask-llm",
    license="MIT",
    author="Florian Magisson",
    author_email="madgic.shrooms@gmail.com",
    description="The easiest way to supercharge your apps with LLM!",
    install_requires=[
        "Jinja2",
        "pillow",
        "langchain>=0.1.10",
        "langchain-community>=0.0.25",
        "langchain-core>=0.1.28",
        "langchain-openai>=0.0.8",
        "langchain-text-splitters>=0.0.1",
        "python-dotenv>=1.0.1",
    ],
)
