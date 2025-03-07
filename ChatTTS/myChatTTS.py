import os, sys

# if sys.platform == "darwin":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from typing import Optional, List
import argparse

import numpy as np

import ChatTTS

from tools.logger import get_logger
from tools.audio import pcm_arr_to_mp3_view
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn

import sounddevice as sd

import numpy as np
import time
import logging
import torch

import wave
import tempfile
import threading
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel as whisper

from openai import OpenAI

logger = get_logger("Command")

AUDIO_BUFFER = 5

client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

def deepseek_infer(role_user):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": role_user},
        ],
        max_tokens=200,
        stream=False
    )

    msg = response.choices[0].message.content
    print(msg)

    return msg


def record_audio(p, device):
    """Record audio from output device and save to temporary WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name
        wave_file = wave.open(filename, "wb")
        wave_file.setnchannels(device["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(device["defaultSampleRate"]))

        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            # print("callback is called")
            wave_file.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=device["maxInputChannels"],
            rate=int(device["defaultSampleRate"]),
            frames_per_buffer=1024,
            input=True,
            # input_device_index=device["index"], # 打开这一行不能正常工作
            stream_callback=callback,
        )

        time.sleep(AUDIO_BUFFER)  # Blocking execution while playing

        stream.stop_stream()
        stream.close()
        wave_file.close()
        print(f"{filename} saved.")
    return filename


# 此函数使用 Whisper 模型对录制的音频进行转录，并输出转录结果。
def whisper_audio(filename, chat):
    """Transcribe audio buffer and display."""
    global has_takeoff
    # segments, info = model.transcribe(filename, beam_size=5, task="translate", language="zh", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
    segments, info = model.transcribe(filename, beam_size=5, language="zh", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
    os.remove(filename)
    # print(f"{filename} removed.")
    for segment in segments:
        if "退出" in segment.text:
            exit()
        if len(segment.text) > 0:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

            # respond_msg = deepseek_infer(segment.text)
            respond_msg = segment.text

            logger.info("ChatTTS Start inference.")
            wavs = chat.infer(respond_msg, use_decoder=True)
            logger.info("ChatTTS Inference completed.")
            
            sd.play(wavs[0], 22400)
            sd.wait()  # 等待播放完成（阻塞模式）

def load_normalizer(chat: ChatTTS.Chat):
    # try to load normalizer
    try:
        chat.normalizer.register("en", normalizer_en_nemo_text())
    except ValueError as e:
        logger.error(e)
    except BaseException:
        logger.warning("Package nemo_text_processing not found!")
        logger.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
        )
    try:
        chat.normalizer.register("zh", normalizer_zh_tn())
    except ValueError as e:
        logger.error(e)
    except BaseException:
        logger.warning("Package WeTextProcessing not found!")
        logger.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
        )


def main(
    spk: Optional[str] = None,
    stream: bool = False,
    source: str = "local",
    custom_path: str = "",
):
    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    load_normalizer(chat)

    is_load = False
    if os.path.isdir(custom_path) and source == "custom":
        is_load = chat.load(source="custom", custom_path=custom_path)
    else:
        is_load = chat.load(source=source)

    if is_load:
        logger.info("ChatTTS Models loaded successfully.")
    else:
        logger.error("ChatTTS Models load failed.")
        sys.exit(1)

    # 初始化音频录入设备
    with pyaudio.PyAudio() as pya:
        # Create PyAudio instance via context manager.
        try:
            # Get default WASAPI info
            wasapi_info = pya.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("Looks like WASAPI is not available on the system. Exiting...")
            sys.exit()

        # Get default WASAPI speakers
        default_speakers = pya.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers["isLoopbackDevice"]:
            for loopback in pya.get_loopback_device_info_generator():
                # Try to find loopback device with same name(and [Loopback suffix]).
                # Unfortunately, this is the most adequate way at the moment.
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print(
                    """
                    Default loopback output device not found.
                    Run `python -m pyaudiowpatch` to check available devices.
                    Exiting...
                    """
                )
                sys.exit()

        print(
            f"Recording from: {default_speakers['name']} ({default_speakers['index']})\n"
        )

        while True:
            print("请说话...")
            filename = record_audio(pya, default_speakers)
            whisper_audio(filename, chat)
            # thread = threading.Thread(target=whisper_audio, args=(filename))
            # thread.start()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

print("Loading Whisper model...")
model_path = r"D:\code-test\tello_drone\large-v3"
# model = whisper("large-v3", device=device, compute_type="float16")
model = whisper(model_size_or_path=model_path, device=device, local_files_only=True)

print("Whisper Model Loaded.")

if __name__ == "__main__":
    r"""
    python -m examples.cmd.run \
        --source custom --custom_path ../../models/2Noise/ChatTTS 你好喲 ":)"
    """
    logger.info("Starting ChatTTS commandline demo...")
    parser = argparse.ArgumentParser(
        description="ChatTTS Command",
        usage='[--spk xxx] [--stream] [--source ***] [--custom_path XXX] "Your text 1." " Your text 2."',
    )
    parser.add_argument(
        "--spk",
        help="Speaker (empty to sample a random one)",
        type=Optional[str],
        default=None,
    )
    parser.add_argument(
        "--stream",
        help="Use stream mode",
        action="store_true",
    )
    parser.add_argument(
        "--source",
        help="source form [ huggingface(hf download), local(ckpt save to asset dir), custom(define) ]",
        type=str,
        default="local",
    )
    parser.add_argument(
        "--custom_path",
        help="custom defined model path(include asset ckpt dir)",
        type=str,
        default="",
    )
    parser.add_argument(
        "texts",
        help="Original text",
        default=["YOUR TEXT HERE"],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger.info(args)
    main(args.spk, args.stream, args.source, args.custom_path)
    logger.info("ChatTTS process finished.")
