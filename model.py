from nicQA import NICQA
import re



if __name__ == "__main__":
    nic_qa = NICQA()
    history = []

    while True:
        question = input("\nASK: ").strip()
        answer = nic_qa.ask(question, history=history)
        history.append({'user_question': question, 'chatBot_answer': answer})
        print(f"\n🔹 **NIC-Assistant:**\n{answer}")