from ask_llm import ask


@ask()
def describe_image(image: str) -> str:
    """
    Describe this image precisely.
    """
