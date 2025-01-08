import concurrent.futures as cf
import glob
import io
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal
import gradio as gr
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
import pymupdf
import re
import requests
from dotenv import load_dotenv

load_dotenv()

# 现在你可以使用 os.getenv() 来获取环境变量
openai_api_key = os.getenv("OPENAI_API_KEY")


# 新增函數，從 API 獲取可用的模型列表
# def fetch_models(api_key):
#     headers = {"Authorization": f"Bearer {api_key}"}
#     response = requests.get("https://api.openai.com/v1/models", headers=headers)

#     if response.status_code == 200:
#         models = response.json()['data']
#         # 返回模型名稱列表，這裡過濾掉非 GPT 模型
#         return [model['id'] for model in models if 'gpt' in model['id']]
#     else:
#         return ["Error fetching models"]

def fetch_models(api_key, api_base=None):
    """
    Fetch the list of models from the given API base.
    """
    # 如果提供了自定义 API base，则使用它；否则使用默认的 OpenAI API base
    base_url = api_base.rstrip("/") + "/models" if api_base else "https://api.openai.com/v1/models"

    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # 发起 GET 请求
        response = requests.get(base_url, headers=headers)

        if response.status_code == 200:
            models = response.json().get('data', [])
            # 返回所有模型的名称列表
            return [model['id'] for model in models]
        else:
            # 如果响应不是 200，返回错误信息
            return [f"Error fetching models: {response.status_code} {response.reason}"]
    except requests.RequestException as e:
        # 捕获请求异常并返回错误信息
        return [f"Error fetching models: {str(e)}"]

def read_readme():
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, "r") as file:
            content = file.read()
            # Use regex to remove metadata enclosed in -- ... --
            content = re.sub(r'--.*?--', '', content, flags=re.DOTALL)
            return content
    else:
        return "README.md not found. Please check the repository for more information."

# Define multiple sets of instruction templates
INSTRUCTION_TEMPLATES = {
################# PODCAST ##################

    "podcast": {
        "intro": """Your task is to take the input text provided and turn it into a lively, engaging, informative podcast dialogue in the style of NPR. 
    The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages.

    We have exactly two speakers in this conversation:
    - speaker-1 (he introduces himself as David)
    - speaker-2 (she introduces herself as Cordelia)

    The conversation must **open** with speaker-1 saying:
    「歡迎來到 David888 Podcast，我是 David...」

    After that, speaker-2 should introduce herself as Cordelia in her first speaking turn.

    Please label each statement or line with speaker-1: or speaker-2: (all lower case, followed by a colon).
    Do not use any other role or bracket placeholders like [Host] or [Guest].

    Don't worry about formatting issues or irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that could be discussed in a podcast.

    Define all terms used carefully for a broad audience of listeners.
    輸出文字為繁體中文，請注意。
    """,
        "text_instructions": "First, carefully read through the input text ...",
        "scratch_pad": """Brainstorm creative ways ...""",
        "prelude": """Now that you have brainstormed ...""",
        "dialog": """Write a very long, engaging, informative podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session.

    Use a **two-speaker** conversational format with exactly:
    - "speaker-1:" (David)
    - "speaker-2:" (Cordelia)

    - The first line must begin with speaker-1: 歡迎來到 David888 Podcast，我是 David...
    - When speaker-2 first speaks, she should introduce herself as Cordelia.

    Alternate turns naturally to simulate an engaging back-and-forth conversation. 
    Do not include bracket placeholders like [Host] or [Guest]; only use speaker-1: or speaker-2: to start each line.

    Design your output to be read aloud, as it will be directly converted into audio. 
    Aim for a long and detailed conversation (around 6000 ~ 8000 words), while still staying on topic and maintaining a fun, accessible style. 
    Make sure to cover all key points and definitions from the input text in a lively, NPR-style format.

    At the end of the dialogue, have the two speakers naturally summarize the main insights and takeaways. 
    Avoid making it sound like an obvious recap; the goal is to gently reinforce the central ideas one last time before signing off.

    請使用繁體中文撰寫。
    """
    },

################# MATERIAL DISCOVERY SUMMARY ##################
    "SciAgents material discovery summary": {
                "intro": """Your task is to take the input text provided and turn it into a lively, engaging conversation between a professor and a student in a panel discussion that describes a new material. The professor acts like Richard Feynman, but you never mention the name.

        The input text is the result of a design developed by SciAgents, an AI tool for scientific discovery that has come up with a detailed materials design.

        Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that could be discussed in a podcast.

        Define all terms used carefully for a broad audience of listeners.
        """,
                "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
                "scratch_pad": """Brainstorm creative ways to discuss the main topics and key points you identified in the material design summary, especially paying attention to design features developed by SciAgents. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

        Keep in mind that your description should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

        Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

        Define all terms used clearly and spend effort to explain the background.

        Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.

        Make sure to make it fun and exciting. You never refer to the podcast, you just discuss the discovery and you focus on the new material design only.
        """,
            "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
    """,
            "dialog": """Write a very long, engaging, informative dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. The presentation must focus on the novel aspects of the material design, behavior, and all related aspects.

    Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience, but make it detailed, logical, and technical so that it has all necessary aspects for listeners to understand the material and its unexpected properties.

    Remember, this describes a design developed by SciAgents, and this must be explicitly stated for the listeners.

    Never use made-up names for the hosts and guests, but make it an engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

    Make the dialogue as long and detailed as possible with great scientific depth, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

    At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.

    The conversation should have around 3000 words. 請用**繁體中文**輸出文稿
    """
        },
################# LECTURE ##################
    "lecture": {
            "intro": """You are Professor Richard Feynman. Your task is to develop a script for a lecture. You never mention your name.

    The material covered in the lecture is based on the provided text.

    Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be covered in the lecture.

    Define all terms used carefully for a broad audience of students.
    """,
            "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
            "scratch_pad": """
    Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

    Keep in mind that your lecture should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

    Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

    Define all terms used clearly and spend effort to explain the background.

    Write your brainstorming ideas and a rough outline for the lecture here. Be sure to note the key insights and takeaways you want to reiterate at the end.

    Make sure to make it fun and exciting.
    """,
            "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
    """,
            "dialog": """Write a very long, engaging, informative script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the students.

    Include clear definitions and terms, and examples.

    Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

    There is only one speaker, you, the professor. Stay on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest lecture you can, while still communicating the key information from the input text in an engaging way.

    At the end of the lecture, naturally summarize the main insights and takeaways from the lecture. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner.

    Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas covered in this lecture one last time before class is over.

    請用**繁體中文**輸出文稿
    """,
        },
################# SUMMARY ##################
        "summary": {
                "intro": """Your task is to develop a summary of a paper. You never mention your name.

        Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.

        Define all terms used carefully for a broad audience.
        """,
                "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
                "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.

        Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.

        Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.

        Make sure to make it engaging and exciting.
        """,
                "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
        """,
                "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.

        Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.

        Include clear definitions and terms, and examples, of all key issues.

        Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

        There is only one speaker, you. Stay on topic and maintaining an engaging flow.

        Naturally summarize the main insights and takeaways from the summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner.

        The summary should have around 1024 words. 請用**繁體中文**輸出文稿
        """,
        },
################# SHORT SUMMARY ##################
        "short summary": {
                "intro": """Your task is to develop a summary of a paper. You never mention your name.

        Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.

        Define all terms used carefully for a broad audience.
        """,
                "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
                "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.

        Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.

        Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.

        Make sure to make it engaging and exciting.
        """,
                "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
        """,
                "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Keep it concise, and use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.

        Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.

        Include clear definitions and terms, and examples, of all key issues.

        Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

        There is only one speaker, you. Stay on topic and maintaining an engaging flow.

        Naturally summarize the main insights and takeaways from the short summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner.

        The summary should have around 256 words. 請用**繁體中文**輸出文稿
        """,
            },

}

# Function to update instruction fields based on template selection
def update_instructions(template):
    return (
        INSTRUCTION_TEMPLATES[template]["intro"],
        INSTRUCTION_TEMPLATES[template]["text_instructions"],
        INSTRUCTION_TEMPLATES[template]["scratch_pad"],
        INSTRUCTION_TEMPLATES[template]["prelude"],
        INSTRUCTION_TEMPLATES[template]["dialog"]
           )


# Define standard values

STANDARD_AUDIO_MODELS = [
    "tts-1",
    "tts-1-hd",
]

STANDARD_VOICES = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
]

class DialogueItem(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2"]

class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]


def get_mp3(text: str, voice: str, audio_model: str, audio_api_key: str) -> bytes:
    """
    Generate audio from text using OpenAI's audio API.
    這裡只使用音頻專用 key。
    """
    client = OpenAI(api_key=audio_api_key)
    with client.audio.speech.with_streaming_response.create(
        model=audio_model,
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()
        

# def get_mp3(text: str, voice: str, audio_model: str, api_key: str = None) -> bytes:
#     client = OpenAI(
#         api_key=api_key or os.getenv("OPENAI_API_KEY"),
#     )

#     with client.audio.speech.with_streaming_response.create(
#         model=audio_model,
#         voice=voice,
#         input=text,
#     ) as response:
#         with io.BytesIO() as file:
#             for chunk in response.iter_bytes():
#                 file.write(chunk)
#             return file.getvalue()


from functools import wraps

def conditional_llm(model, api_base=None, api_key=None):
    """
    Conditionally apply the @llm decorator based on the api_base parameter.
    If api_base is provided, it applies the @llm decorator with api_base.
    Otherwise, it applies the @llm decorator without api_base.
    """
    def decorator(func):
        if api_base:
            return llm(model=model, api_base=api_base)(func)
        else:
            return llm(model=model, api_key=api_key)(func)
    return decorator

def generate_audio(
    files: list,
    openai_api_key: str,       
    openai_audio_api_key: str, 
    text_model: str,
    intro_instructions: str = '',
    text_instructions: str = '',
    scratch_pad_instructions: str = '',
    prelude_dialog: str = '',
    podcast_dialog_instructions: str = '',
    edited_transcript: str = None,
    user_feedback: str = None,
    original_text: str = None,
    api_base_value: str = None,
    speaker_1_voice: str = "onyx",
    speaker_2_voice: str = "nova",
) -> tuple[bytes, str]:
    print("Starting generate_audio...")
    logger.info("Starting generate_audio...")

    combined_text = original_text or ""
    if not combined_text:
        print("Extracting text from uploaded PDFs...")
        logger.info("Extracting text from uploaded PDFs...")
        for file in files:
            doc = pymupdf.open(file.name)
            for page in doc:
                combined_text += page.get_text() + "\n\n"

    print("Text extraction completed.")
    logger.info("Text extraction completed.")

    print("Calling generate_dialogue_via_requests...")
    logger.info("Calling generate_dialogue_via_requests...")

    # ★ 傳入 pdf_text=combined_text
    dialogue = generate_dialogue_via_requests(
        pdf_text=combined_text,  # ← 新增
        intro_instructions=intro_instructions,
        text_instructions=text_instructions,
        scratch_pad_instructions=scratch_pad_instructions,
        prelude_dialog=prelude_dialog,
        podcast_dialog_instructions=podcast_dialog_instructions,
        model=text_model,
        llm_api_key=openai_api_key,
        api_base=api_base_value,
        edited_transcript=edited_transcript,
        user_feedback=user_feedback
    )

    print(f"Dialogue generation result: {dialogue}")
    logger.info(f"Dialogue generation result: {dialogue}")

    audio = b""
    transcript = ""
    characters = 0

    # 多執行緒加速 TTS
    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for line in dialogue.splitlines():
            raw_line = line.strip()
            if not raw_line:
                continue

            # 預設使用 speaker 1
            voice_to_use = speaker_1_voice
            line_text = raw_line

            # 如果行首包含 "speaker-1:" 或 "speaker-2:"
            # 這裡你可以根據實際 LLM 產生的文字去調整判斷關鍵字
            if raw_line.lower().startswith("speaker-1:"):
                voice_to_use = speaker_1_voice
                line_text = raw_line.split(":", 1)[1].strip()  # 取冒號後面的內容
            elif raw_line.lower().startswith("speaker-2:"):
                voice_to_use = speaker_2_voice
                line_text = raw_line.split(":", 1)[1].strip()

            # 提交到執行緒池
            future = executor.submit(
                get_mp3,
                line_text,
                voice_to_use,
                "tts-1",                # 你的音頻模型
                openai_audio_api_key
            )
            futures.append((future, raw_line))  # raw_line 僅作為 transcript 顯示

            characters += len(line_text)

        # 收集結果
        for future, raw_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += raw_line + "\n\n"

    return audio, transcript

def generate_dialogue_via_requests(
    pdf_text: str,                # ← 新增
    intro_instructions: str,
    text_instructions: str,
    scratch_pad_instructions: str,
    prelude_dialog: str,
    podcast_dialog_instructions: str,
    model: str,
    llm_api_key: str,
    api_base: str,
    edited_transcript: str = None,
    user_feedback: str = None
) -> str:
    """
    Generate dialogue by making a direct request to the LLM API.
    """

    # ★ 在 merged_content 中插入 PDF 內容
    merged_content = f"""
以下是從 PDF 中擷取的文字內容，請參考並納入對話:
================================
{pdf_text}
================================

{intro_instructions}
{text_instructions}

<scratchpad>
{scratch_pad_instructions}
</scratchpad>

{prelude_dialog}

<podcast_dialogue>
{podcast_dialog_instructions}
</podcast_dialogue>

{edited_transcript or ""}
{user_feedback or ""}
"""

    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": merged_content
            }
        ],
        "temperature": 0.7,
        "max_tokens": 900048
    }

    base_url = api_base.rstrip("/")
    url = f"{base_url}/chat/completions"

    print(f"Calling LLM API with model: {model}, API base: {base_url}")
    logger.info(f"Calling LLM API with model: {model}, API base: {base_url}")

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"API Response: {result}")
        logger.info(f"API Response: {result}")

        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        logger.error(f"Error during API call: {e}")
        return f"Error: {e}"





    # Generate audio from the transcript
    audio = b""
    transcript = ""
    characters = 0

    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for line in llm_output.dialogue:
            transcript_line = f"{line.speaker}: {line.text}"
            voice = speaker_1_voice if line.speaker == "speaker-1" else speaker_2_voice
            future = executor.submit(get_mp3, line.text, voice, audio_model, openai_api_key)
            futures.append((future, transcript_line))
            characters += len(line.text)

        for future, transcript_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += transcript_line + "\n\n"

    logger.info(f"Generated {characters} characters of audio")

    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    # Use a temporary file -- Gradio's audio component doesn't work with raw bytes in Safari
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

    return temporary_file.name, transcript, combined_text

def validate_and_generate_audio(
    files,
    openai_api_key,            # Gemini (LLM) 的 Key
    openai_audio_api_key,      # OpenAI TTS 的 Key
    text_model,
    audio_model,
    speaker_1_voice,
    speaker_2_voice,
    api_base_value,
    intro_instructions,
    text_instructions,
    scratch_pad_instructions,
    prelude_dialog,
    podcast_dialog_instructions,
    edited_transcript,
    user_feedback
):
    print("Starting validate_and_generate_audio...")
    logger.info("Starting validate_and_generate_audio...")

    if not files:
        print("No files uploaded.")
        logger.warning("No files uploaded.")
        return None, None, None, "Please upload at least one PDF file before generating audio."

    try:
        print("Calling generate_audio with the provided inputs...")
        logger.info("Calling generate_audio with the provided inputs...")

        # 這裡同時把 openai_api_key (LLM) 與 openai_audio_api_key (TTS) 傳進去
        audio_file, transcript = generate_audio(
            files=files,
            openai_api_key=openai_api_key,
            openai_audio_api_key=openai_audio_api_key,
            text_model=text_model,
            intro_instructions=intro_instructions,
            text_instructions=text_instructions,
            scratch_pad_instructions=scratch_pad_instructions,
            prelude_dialog=prelude_dialog,
            podcast_dialog_instructions=podcast_dialog_instructions,
            edited_transcript=edited_transcript,
            user_feedback=user_feedback,
            api_base_value=api_base_value
        )

        print("Audio generation completed.")
        logger.info("Audio generation completed.")

        return audio_file, transcript, None, None

    except Exception as e:
        print(f"Error during audio generation: {e}")
        logger.error(f"Error during audio generation: {e}")
        return None, None, None, str(e)



def edit_and_regenerate(edited_transcript, user_feedback, *args):
    # Replace the original transcript and feedback in the args with the new ones
    #new_args = list(args)
    #new_args[-2] = edited_transcript  # Update edited transcript
    #new_args[-1] = user_feedback  # Update user feedback
    return validate_and_generate_audio(*new_args)

# New function to handle user feedback and regeneration
def process_feedback_and_regenerate(feedback, *args):
    # Add user feedback to the args
    new_args = list(args)
    new_args.append(feedback)  # Add user feedback as a new argument
    return validate_and_generate_audio(*new_args)



with gr.Blocks(title="PDF to Podcast", css="""
    #header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px;
        background-color: transparent;
        border-bottom: 1px solid #ddd;
    }
    #title {
        font-size: 24px;
        margin: 0;
    }
    #logo_container {
        width: 200px;
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #logo_image {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    #main_container {
        margin-top: 20px;
    }
""") as demo:

    with gr.Row(elem_id="header"):
        with gr.Column(scale=4):
            gr.Markdown("# Oli家：把你的PDF文件轉成Podcast聲音廣播節目。Convert PDFs into an audio podcast, lecture, summary and others\n\nFirst, upload one or more PDFs, select options, then push Generate Podcast.\n首先，上傳一個或多個 PDF，選擇選項，然後按下「Generate Podcast」按鈕。\nYou can also select a variety of custom option and direct the way the result is generated.\n您還可以選擇各種自訂選項。", elem_id="title")
        with gr.Column(scale=1):
            gr.HTML('''
                <div id="logo_container">
                    <img src="https://huggingface.co/spaces/lamm-mit/PDF2Audio/resolve/main/logo.png" id="logo_image" alt="Logo">
                </div>
            ''')

    submit_btn = gr.Button("產生Podcast聲音 | Generate Podcast", elem_id="submit_btn")

    with gr.Row(elem_id="main_container"):
        # 左邊的欄位
        with gr.Column(scale=2):
            # 新增 Custom API Base，並設置為 OpenAI API Key 的上方，且增加預設值
            # 文件上傳區域，支援拖曳
            files = gr.Files(
                label="PDFs",
                file_types=[".pdf"],       # 限制文件類型為 PDF
                file_count="multiple",     # 支援多文件上傳
                interactive=True           # 確保拖曳功能可用
            )

            # api_base_input = gr.Textbox(
            #     label="Custom API Base",
            #     placeholder="https://api.openai.com/v1",
            #     value="https://api.openai.com/v1",
            # )
            
            api_base = gr.Textbox(
                label="Custom API Base",
                placeholder="https://api.openai.com/v1/models",  # 預設值顯示
                value="https://api.openai.com/v1/models",        # 預設實際值
                info="If you are using a custom or local model, provide the API base URL here.",
            )

            openai_api_key = gr.Textbox(
                label="LLM API Key",
                visible=True,
                placeholder="這行要填可以用 LLM api key",
                type="password"
            )


            # 創建一個空的下拉框，選項會在按鈕點擊時從 API 獲取並動態填充
            text_model = gr.Dropdown(
                label="Text Generation Model",
                choices=[],  # 初始為空
                value="",    # 初始值設為空
                allow_custom_value=True,  # 允許自定義值
                info="Select the model to generate the dialogue text.",
            )

            # 新增一個按鈕來動態獲取模型列表
            fetch_button = gr.Button("獲取模型列表 | Fetch Models")

            # 當按鈕被點擊時，從 API 獲取模型並更新下拉選單
            fetch_button.click(
                fn=lambda api_key, api_base: gr.update(choices=fetch_models(api_key, api_base)),
                inputs=[openai_api_key, api_base],  # 傳入 OpenAI API Key 和 Custom API Base
                outputs=[text_model]
            )


            # Key specifically for OpenAI audio generation
            openai_audio_api_key = gr.Textbox(
                label="OpenAI Audio API Key",
                visible=True,
                placeholder="Enter your OpenAI API key for audio generation",
                type="password"
            )

            # 音頻模型下拉框依然是靜態選擇
            audio_model = gr.Dropdown(
                label="Audio Generation Model",
                choices=STANDARD_AUDIO_MODELS,
                value="tts-1",
                info="Select the model to generate the audio.",
            )

            # Speaker 1 的聲音選擇
            speaker_1_voice = gr.Dropdown(
                label="Speaker 1 Voice",
                choices=STANDARD_VOICES,
                value="onyx",
                info="Select the voice for Speaker 1.",
            )

            # Speaker 2 的聲音選擇
            speaker_2_voice = gr.Dropdown(
                label="Speaker 2 Voice",
                choices=STANDARD_VOICES,
                value="nova",
                info="Select the voice for Speaker 2.",
            )


        # 右邊的欄位
        with gr.Column(scale=3):
            audio_output = gr.Audio(label="Audio", format="mp3", interactive=False, autoplay=False)
            transcript_output = gr.Textbox(label="Transcript", lines=20, show_copy_button=True)
            original_text_output = gr.Textbox(label="Original Text", lines=10, visible=False)
            error_output = gr.Textbox(visible=False)

            use_edited_transcript = gr.Checkbox(label="Use Edited Transcript", value=False)
            edited_transcript = gr.Textbox(label="Edit Transcript Here", lines=20, visible=False, show_copy_button=True, interactive=False)

            user_feedback = gr.Textbox(label="Provide Feedback or Notes", lines=10)
            regenerate_btn = gr.Button("Regenerate Podcast with Edits and Feedback")

        # 中間部分輸入選項區域放置在下方
        with gr.Column(scale=3):
            template_dropdown = gr.Dropdown(
                label="Instruction Template",
                choices=list(INSTRUCTION_TEMPLATES.keys()),
                value="podcast",
                info="Select the instruction template to use. You can also edit any of the fields for more tailored results.",
            )
            intro_instructions = gr.Textbox(
                label="Intro Instructions",
                lines=10,
                value=INSTRUCTION_TEMPLATES["podcast"]["intro"],
                info="Provide the introductory instructions for generating the dialogue.",
            )
            text_instructions = gr.Textbox(
                label="Standard Text Analysis Instructions",
                lines=10,
                placeholder="Enter text analysis instructions...",
                value=INSTRUCTION_TEMPLATES["podcast"]["text_instructions"],
                info="Provide the instructions for analyzing the raw data and text.",
            )
            scratch_pad_instructions = gr.Textbox(
                label="Scratch Pad Instructions",
                lines=15,
                value=INSTRUCTION_TEMPLATES["podcast"]["scratch_pad"],
                info="Provide the scratch pad instructions for brainstorming presentation/dialogue content.",
            )
            prelude_dialog = gr.Textbox(
                label="Prelude Dialog",
                lines=5,
                value=INSTRUCTION_TEMPLATES["podcast"]["prelude"],
                info="Provide the prelude instructions before the presentation/dialogue is developed.",
            )
            podcast_dialog_instructions = gr.Textbox(
                label="Podcast Dialog Instructions",
                lines=20,
                value=INSTRUCTION_TEMPLATES["podcast"]["dialog"],
                info="Provide the instructions for generating the presentation or podcast dialogue.",
            )

    def update_edit_box(checkbox_value):
        return gr.update(interactive=checkbox_value, lines=20 if checkbox_value else 20, visible=True if checkbox_value else False)

    use_edited_transcript.change(fn=update_edit_box, inputs=[use_edited_transcript], outputs=[edited_transcript])
    template_dropdown.change(fn=update_instructions, inputs=[template_dropdown], outputs=[intro_instructions, text_instructions, scratch_pad_instructions, prelude_dialog, podcast_dialog_instructions])

    submit_btn.click(
        fn=validate_and_generate_audio,
        inputs=[
            files,
            openai_api_key,           # LLM (Gemini) Key
            openai_audio_api_key,     # Audio (OpenAI) Key
            text_model,
            audio_model,
            speaker_1_voice,
            speaker_2_voice,
            api_base,
            intro_instructions,
            text_instructions,
            scratch_pad_instructions,
            prelude_dialog,
            podcast_dialog_instructions,
            edited_transcript,
            user_feedback
        ],
        outputs=[audio_output, transcript_output, original_text_output, error_output],
    )
 





    regenerate_btn.click(
        fn=lambda use_edit, edit, *args: validate_and_generate_audio(*args[:12], edit if use_edit else "", *args[12:]),
        inputs=[use_edited_transcript, edited_transcript, files, openai_api_key, text_model, audio_model, speaker_1_voice, speaker_2_voice, api_base, intro_instructions, text_instructions, scratch_pad_instructions, prelude_dialog, podcast_dialog_instructions, user_feedback, original_text_output],
        outputs=[audio_output, transcript_output, original_text_output, error_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
