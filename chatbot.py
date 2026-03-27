import requests
import re

# MCP server endpoint
MCP_SERVER_URL = "http://localhost:8000/query"  # Update if hosted elsewhere


def ask_question(question, top_k=3):
    """
    Sends a question to the MCP server and returns the answer.
    """
    payload = {"query": question, "top_k": top_k}

    try:
        response = requests.post(MCP_SERVER_URL, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            # Merge all top_k results into one string
            if "answer" in data:
                answer_text = data["answer"]
            elif "results" in data:
                # Some MCP servers may return 'results'
                answer_text = " ".join(r.get("text", "") for r in data["results"])
            else:
                answer_text = "No answer found."
            return answer_text.strip()
        else:
            return f"Error: Server returned status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to MCP server. {e}"


def format_answer(answer, max_sentence_length=300):
    """
    Splits answer into complete sentences and trims overly long ones.
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    formatted_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Trim if sentence is too long
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length].rstrip() + "…"
        formatted_sentences.append(sentence)

    return formatted_sentences


def main():
    print("=== RAG Chatbot for UntMart Website ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Optional: allow user to specify top_k like "question | 3"
        if "|" in user_input:
            question, top_k_str = map(str.strip, user_input.split("|", 1))
            try:
                top_k = int(top_k_str)
            except ValueError:
                top_k = 3
        else:
            question = user_input
            top_k = 2

        answer = ask_question(question, top_k=top_k)
        sentences = format_answer(answer)

        print("Bot:")
        for s in sentences:
            print(s)
        print("-" * 50)


if __name__ == "__main__":
    main()