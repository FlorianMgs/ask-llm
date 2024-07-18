from ask_llm import ask


@ask()
def conversation(instruction: str) -> str:
    """
    {% chat %}
        {% message system %}
            You are an helpful assistant that can answer all the user questions.
        {% endmessage %}
        {% message ai %}
            Hello, Arnaud! How can I help you today?
        {% endmessage %}
        {% message human %}
            {{ instruction }}
        {% endmessage %}
    {% endchat %}
    Answer in CAPS LOCK
    """
