import os
import openai
import time
from openai import OpenAI

DATASETS_DIR = "datasets"
FILE_NAME = "test_dataset.json"

openai.api_key = "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"
assistant_id = "asst_1QTnFga3XfjYvGkM1vWSImc0"

client = OpenAI(api_key="Api key")

def assitent_creation():
    file = client.files.create(
        file=open("datasets/test_dataset.json", "rb"),
        purpose="assistants"
    )

    assistant = client.beta.assistants.create(
        name="Trader",
        instructions="Здесь инструкции типо ты очень крутой трейдер ",
        tools=[{"type": "code_interpreter"}],
        model="gpt-3.5-turbo-1106",
        file_ids=[file.id]
    )
    return assistant.id


def create_thread(ass_id, prompt):
    thread = openai.beta.threads.create()
    message = openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    # run
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ass_id,
    )
    return run.id, thread.id


def check_status(run_id, thread_id):
    run = openai.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return run.status


def main():
    prompt = "Try to suppose usdt changes after this news: Snapshot. Taken. A snapshot of 200M NFTs has just been taken on Ethereum and JumpNet. The NFTs of the two biggest ERC-1155 smart contracts will migrate to Enjin Blockchain.Learn more: https://t.co/1hb0C2Frlt. Once migrated, you’ll be able to: Trade your NFTs for free… https://t.co/TRvtm0NkrJ https://t.co/3IK8OyCPnM"
    run_id, thread_id = create_thread(ass_id=assistant_id, prompt=prompt)

    status = check_status(run_id, thread_id)
    while status != "completed":
        status = check_status(run_id, thread_id)
        time.sleep(2)
    response = openai.beta.threads.messages.list(
        thread_id=thread_id
    )
    if response.data:
        print(response.data[0].content[0].text.value)


if __name__ == "__main__":
    main()
