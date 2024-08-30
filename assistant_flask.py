import base64
from threading import Lock, Thread

import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from flask import Flask, Response, render_template, jsonify
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()

class WebcamStream:
    def __init__(self):     
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer).decode('utf-8')

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.last_prompt = None
        self.last_response = None

    def answer(self, prompt, image):
        if not prompt:
            return

        self.last_prompt = prompt
        print("Prompt:", prompt)

        # Encode the image as a JPEG
        _, buffer = cv2.imencode(".jpeg", image)
        
        # Convert the buffer to a base64 string
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image_base64},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        self.last_response = response
        print("Response:", response)
        
        if response:
            self._tts(response)
               
    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        # SYSTEM_PROMPT = """
        # You are being used to power a video assistant that will use the chat history and the image 
        # provided by the user to answer its questions. Wait for the user prompt
        # and greet them for the first time.

        # Do not use any emoticons or emojis. Do not use any special charecters.

        # Be friendly and helpful. Show some personality. Do not be too formal.
        # """

        SYSTEM_PROMPT = """
        You are being used to power a video assistant and you have knowledge on celebraties that will use the chat history and the image 
        provided by the user to answer its questions. Wait for the user prompt
        and greet them for the first time.

        recognize the actors in the image and answer the questions based on the actors in the image.

        recognize the image who is speaking with you and remember his name and whenever he asks respond accordingly

        Do not use any emoticons or emojis. Do not use any special charecters.

        Be friendly and helpful. Show some personality. Do not be too formal.

        weather outside is too rainy so, only if user says he is going out, check the images for knowing type of clothes and if they are not wollen clothes and raincoat tell him to wear wollen clother and raincap or else he may get cold 
        """        

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

webcam_stream = WebcamStream().start()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
# model = ChatOpenAI(model="gpt-4o")

assistant = Assistant(model)

def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read())  # No change needed here

    except UnknownValueError:
        print("There was an error processing the audio.")

recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

app = Flask(__name__)

@app.route('/')
def index():
    # Serve the HTML page
    return render_template('index.html')

def generate_frames():
    while True:
        frame = webcam_stream.read()
        _, buffer = imencode('.jpeg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/get_conversation')
def get_conversation():
    return jsonify({
        "prompt": assistant.last_prompt,
        "response": assistant.last_response
    
    })


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)