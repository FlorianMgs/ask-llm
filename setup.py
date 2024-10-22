from setuptools import setup, find_packages

setup(
    name="ask-llm",
    version="0.1.183",
    packages=find_packages(),
    url="https://github.com/FlorianMgs/ask-llm",
    license="MIT",
    author="Florian Magisson",
    author_email="madgic.shrooms@gmail.com",
    description="The easiest way to supercharge your apps with LLM!",
    install_requires=[
        "Jinja2>=3.1.3",
        "pillow>=10.2.0",
        "langchain>=0.3.2",
        "langchain-community>=0.3.1",
        "langchain-core>=0.3.9",
        "langchain-openai>=0.2.0",
        "langchain-anthropic>=0.2.3",
        "python-dotenv>=1.0.1",
    ],
)
