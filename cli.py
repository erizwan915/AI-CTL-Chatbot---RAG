from myollama import chatbot, load_expanded_chunks, build_index

def run_cli():
    messages = [{
    "role": "system",
    "content": (
        "You are an administrative assistant for the Knox College Center for Teaching and Learning (CTL). "
        "Your job is to help students understand when and where tutors are available, what subjects they cover, "
        "and how students can schedule appointments with them. "
        "Respond in **clear, structured Markdown format** for readability — use:\n"
        "- Bullet points for multiple tutors or schedules\n"
        "- Bold names for tutors\n"
        "- Blank lines between different entries\n"
        "- Hyperlinked calendar URLs if available\n\n"
        "Keep your tone warm, clear, and helpful."
    )
}]


    print("LLaMA3.2 RAG Chatbot — type 'exit' to quit\n")

    chunks = load_expanded_chunks("expanded_tutor_chunks.csv")
    index, embeddings, chunks, embed_model = build_index(chunks)

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        reply = chatbot(user_input, messages, index, chunks, embeddings, embed_model)
        print("AI:", reply)

if __name__ == "__main__":
    run_cli()
