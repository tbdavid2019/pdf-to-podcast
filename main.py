import concurrent.futures as cf
import glob
import io
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal

import gradio as gr
import sentry_sdk
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type
from dotenv import load_dotenv
load_dotenv()
# 加入以下代碼以驗證 API KEY 是否已正確加載
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    logger.error("OPENAI_API_KEY not loaded")
else:
    logger.info(f"OPENAI_API_KEY loaded: {openai_key[:5]}***")

sentry_sdk.init(os.getenv("SENTRY_DSN"))

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class DialogueItem(BaseModel):
    text: str
    speaker: Literal["female-1", "male-1", "female-2"]

    @property
    def voice(self):
        return {
            "female-1": "nova",
            "male-1": "onyx",
            "female-2": "shimmer",
        }[self.speaker]


class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]


def get_mp3(text: str, voice: str, api_key: str = None) -> bytes:
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()


def generate_audio(file: str, openai_api_key: str = None) -> bytes:

    if not os.getenv("OPENAI_API_KEY", openai_api_key):
        raise gr.Error("OpenAI API key is required")

    with Path(file).open("rb") as f:
        reader = PdfReader(f)
        text = "\n\n".join([page.extract_text() for page in reader.pages])

    @retry(retry=retry_if_exception_type(ValidationError))
    @llm(
        model="gpt-4o",
    )
    def generate_dialogue(text: str) -> Dialogue:
        """
        Your task is to take the input text provided and turn it into an engaging, informative podcast dialogue. The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points and interesting facts that could be discussed in a podcast.

        Here is the input text you will be working with:

        <input_text>
        {text}
        </input_text>

        First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for an audio podcast.

    <scratchpad>
        Take some time to carefully identify not only the main topics but also any subtle themes or interesting side notes that could be expanded upon. For each main topic, think about how you can dive deeper and provide additional context, background information, or contrasting perspectives.

        Be creative in finding ways to connect different points of discussion. Consider using analogies, metaphors, or cultural references to make the content more engaging and relatable. You can also include relevant historical or contemporary events, or even explore "what if" scenarios to add depth to the conversation.

        Think about including different formats in the dialogue—such as a Q&A section, hypothetical debates between the host and guest, or even brief role-playing segments where the speakers act out certain ideas. These varied formats can help keep the dialogue dynamic and interesting.

        Additionally, come up with open-ended, thought-provoking questions to guide the conversation. These questions should encourage deeper exploration of the topic and invite the guest to provide personal insights, real-world applications, or speculative answers.

        Don’t hesitate to integrate humor, surprising facts, or anecdotes that could lighten the mood and make the dialogue more entertaining for listeners. Think about ways to transition smoothly between serious and lighter moments to create a balanced and engaging flow.

        At the end of the brainstorming session, list at least three key takeaways or conclusions that you want to reinforce in the final dialogue. These should be seamlessly woven into the conversation, leading to a natural conclusion.
    </scratchpad>

        Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.

        <podcast_dialogue>
        Write your engaging, informative podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience. Use made-up names for the hosts and guests to create a more engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

        Make the dialogue as long and detailed as possible, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

        At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.

        Make the dialogue detailed and expansive. Try to include at least three key points or examples, expanding on each one in depth. Ensure that the dialogue flows naturally and each segment contributes to a longer overall conversation.
 
               請輸出講稿約 3,000 characters，輸出文稿為__繁體中文__，請注意。
        </podcast_dialogue>
        """

    llm_output = generate_dialogue(text)

    audio = b""
    transcript = ""

    characters = 0

    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for line in llm_output.dialogue:
            transcript_line = f"{line.speaker}: {line.text}"
            future = executor.submit(get_mp3, line.text, line.voice, openai_api_key)
            futures.append((future, transcript_line))
            characters += len(line.text)

        for future, transcript_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += transcript_line + "\n\n"

    logger.info(f"Generated {characters} characters of audio")

    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    # we use a temporary file because Gradio's audio component doesn't work with raw bytes in Safari
    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    temporary_file.write(audio)
    temporary_file.close()

    # Delete any files in the temp directory that end with .mp3 and are over a day old
    for file in glob.glob(f"{temporary_directory}*.mp3"):
        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60:
            os.remove(file)

    return temporary_file.name, transcript


demo = gr.Interface(
    title="PDF to Podcast",
    description=Path("description.md").read_text(),
    fn=generate_audio,
    examples=[[str(p)] for p in Path("examples").glob("*.pdf")],
    inputs=[
        gr.File(
            label="PDF",
        ),
        gr.Textbox(
            label="OpenAI API Key",
            visible=True,  # 確保它始終可見
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
        gr.Textbox(label="Transcript"),
    ],
    allow_flagging="never",
    clear_btn=None,
    head=os.getenv("HEAD", "") + Path("head.html").read_text(),
    cache_examples="lazy",
    api_name=False,
)


demo = demo.queue(
    max_size=20,
    default_concurrency_limit=20,
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    demo.launch(show_api=False)
