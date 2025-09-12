# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("armitaraz/chatgpt-reddit")

# print("Path to dataset files:", path)


# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("mowglii/twitch-chat-test-data")

# print("Path to dataset files:", path)

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("kreeshrajani/3k-conversations-dataset-for-chatbot")

# print("Path to dataset files:", path)



import json
import csv

input_csv = "data/Conversation.csv"
output_jsonl = "data/Conversation.jsonl"

with open(input_csv, newline='', encoding="utf-8") as csvfile, \
     open(output_jsonl, "w", encoding="utf-8") as jsonlfile:

    reader = csv.DictReader(csvfile)
    for row in reader:
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        if question and answer:  # skip empty messages
            json_line = {"question": question, "answer": answer}
            json.dump(json_line, jsonlfile)
            jsonlfile.write("\n")




# import csv
# import json
# import random

# # Longer, Twitch/chat-style assistant responses
# assistant_phrases = [
#     "Haha chat, I see you! That's classic.",
#     "Whoa, that hits hard! Gotta love this community.",
#     "Big respect, everyone in the chat feels that!",
#     "Wow, chat, absolutely insane!",
#     "Haha, that’s hilarious, couldn’t have said it better!",
#     "Yooo, that’s epic! Keep it going, chat!",
#     "Feels like a movie plot, honestly.",
#     "OMG chat, can't believe that just happened!",
#     "Hah, the energy is unmatched, love it!",
#     "This is wild, I’m crying laughing!"
# ]

# def generate_assistant_response(user_msg):
#     return f"{random.choice(assistant_phrases)} ({user_msg})"

# # Input CSV
# csv_file = "data/twitch_data.csv"
# # Output JSONL
# jsonl_file = "data/twitch_dataset_test.jsonl"

# with open(csv_file, encoding="utf-8") as f, open(jsonl_file, "w", encoding="utf-8") as out_f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         user_msg = row["message"].strip()
#         if not user_msg:
#             continue
        
#         # Create JSONL chat message
#         data = {
#             "messages": [
#                 {"role": "user", "content": user_msg},
#                 {"role": "assistant", "content": generate_assistant_response(user_msg)}
#             ]
#         }
#         out_f.write(json.dumps(data, ensure_ascii=False) + "\n")

# print(f"✅ JSONL dataset ready: {jsonl_file}")
