from setuptools import setup, find_packages

setup(
    name="ask-llm",
    version="0.1.14",
    packages=find_packages(),
    url="https://github.com/FlorianMgs/ask-llm",
    license="MIT",
    author="Florian Magisson",
    author_email="madgic.shrooms@gmail.com",
    description="The easiest way to supercharge your apps with LLM!",
    install_requires=[
        "Jinja2",
        "pillow",
        "langchain>=0.2.9",
        "langchain-community>=0.2.7",
        "langchain-core>=0.2.21",
        "langchain-openai>=0.1.17",
        "python-dotenv>=1.0.1",
    ],
)
