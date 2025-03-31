import aiml
import os

# Initialize AIML Kernel
aiml_kernel = aiml.Kernel()

# Load AIML rules from aiml_rules directory
aiml_rules_path = "aiml_rules"
os.chdir(aiml_rules_path)
aiml_kernel.learn("*.aiml")
os.chdir("..")  # Move back to main directory

# Run the AIML-based chatbot
if __name__ == "__main__":
    print("AIML Chatbot: Hello! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        user_input = user_input.lower()
        if user_input.lower() in ["exit", "quit"]:
            print("AIML Chatbot: Goodbye!")
            break

        # Get response from AIML
        aiml_response = aiml_kernel.respond(user_input)

        if aiml_response.strip():
            print(f"AIML Chatbot: {aiml_response}")
        else:
            print("AIML Chatbot: Sorry, I don't understand that.")