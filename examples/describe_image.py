from ask_llm import ask


@ask(model_name="claude-3-5-sonnet-20240620")
def describe_image(image: str) -> str:
    """
    Describe this image precisely.
    """
