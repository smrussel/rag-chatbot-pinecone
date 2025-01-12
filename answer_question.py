from utils import get_embedding
from flask import jsonify
from config import *
from flask import current_app

import openai

from config import *

openai.api_key = OPENAI_API_KEY

TOP_K = 5


def get_unique_values(input_list):
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

def get_answer_from_files(question, session_id, pinecone_index):
    # logging.info(f"Getting answer for question: {question}")
    # print(f"Getting answer for question: {question}")

    search_query_embedding = get_embedding(question, EMBEDDINGS_MODEL)

    try:
        query_response = pinecone_index.query(
            namespace=session_id,
            top_k=TOP_K,
            # include_values=False,
            include_metadata=True,
            vector=search_query_embedding,
        )
        # logging.info(
        #     f"[get_answer_from_files] received query response from Pinecone: {query_response}")
        # print(
            # f"[get_answer_from_files] received query response from Pinecone: {query_response}")

        files_string = ""
        file_text_dict = current_app.config["file_text_dict"]
        # print(file_text_dict)
        sources = []
        for i in range(len(query_response.matches)):
            result = query_response.matches[i]
            file_chunk_id = result.id
            score = result.score
            filename = result.metadata["filename"]
            sources.append(filename)
            # file_text = file_text_dict.get(file_chunk_id)
            file_text = result.metadata["text"]
            file_string = f"###\n\"{filename}\"\n{file_text}\n"
            if score < COSINE_SIM_THRESHOLD and i > 0:
                logging.info(
                    f"[get_answer_from_files] score {score} is below threshold {COSINE_SIM_THRESHOLD} and i is {i}, breaking")
                break
            files_string += file_string
        
        # Note: this is not the proper way to use the ChatGPT conversational format, but it works for now
        # messages = [
        #     {
        #         "role": "system",
        #         "content": f"Given a question, try to answer it using the content of the file extracts below, and if you cannot answer, or find " \
        #         f"a relevant file, just output \"I couldn't find the answer to that question in your files.\".\n\n" \
        #         f"If the answer is not contained in the files or if there are no file extracts, respond with \"I couldn't find the answer " \
        #         f"to that question in your files.\" If the question is not actually a question, respond with \"That's not a valid question.\"\n\n" \
        #         f"In the cases where you can find the answer, first give the answer. Then explain how you found the answer from the source or sources, " \
        #         f"and use the exact filenames of the source files you mention. Do not make up the names of any other files other than those mentioned "\
        #         f"in the files context. Give the answer in markdown format." \
        #         f"Use the following format:\n\nQuestion: <question>\n\nFiles:\n<###\n\"filename 1\"\nfile text>\n<###\n\"filename 2\"\nfile text>...\n\n"\
        #         f"Answer: <answer or \"I couldn't find the answer to that question in your files\" or \"That's not a valid question.\">\n\n" \
        #         f"Question: {question}\n\n" \
        #         f"Files:\n{files_string}\n" \
        #         f"Answer:"
        #     },
        # ]

        messages = [
            {
                "role": "system",
                "content": f"""
                You are a helpful AI assistant named jervis . The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end. 
                If you don't know the answer, just say you don't know. Do NOT try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer
                questions that are related to the context. Use as much detail as possible when responding. 
                context: {files_string} \n\n
                Question: {question} \n\n
                Answer:
                """
            },
        ]
        
        
        response = openai.ChatCompletion.create(
            messages=messages,
            model=GENERATIVE_MODEL,
            max_tokens=1000,
            temperature=0,
        )

        choices = response["choices"]  # type: ignore
        answer = choices[0].message.content.strip()
        # print(answer)
        sources = get_unique_values(sources)
        # logging.info(f"[get_answer_from_files] answer: {answer}")
        print(f"[get_answer_from_files] answer: {answer}")
        return jsonify({"success": True,"answer": answer,"sources":sources})

    except Exception as e:
        # logging.info(f"[get_answer_from_files] error: {e}")
        print(f"[get_answer_from_files] error: {e}")
        return jsonify({"success": False,"answer": "Sorry something went wrong, Please try again."})
