import os
import azure.cognitiveservices.speech as speechsdk
import openai
import time
from mpi4py import MPI


# This example requires environment variables named "OPEN_AI_KEY" and "OPEN_AI_ENDPOINT"
# Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/
openai.api_key = ''
openai.api_base =  ''
openai.api_type = 'azure'
openai.api_version = '2023-07-01-preview'

# This will correspond to the custom name you chose for your deployment when you deployed a model.
deployment_id='' 

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription='', region='')


# Prompts Azure OpenAI with a request and synthesizes the response.
def ask_openai(comm):
    sentance = 0
    while True:
        prompt = comm.recv(source=0)

        messages = [{"role":"system",
                    "content":"You are an English-Chinese translator, your task is to accurately translate text between the two languages. Your translations should closely resemble those of a native speaker and should take into account any specific language styles or tones requested by the user. Please review and revise your answers carefully before submitting. Do not answer the users. Do not include anything other than the translated content in your output."},
                    {"role":"user","content":"{}".format(prompt)}]

        print('RECOGNIZED: {}'.format(prompt))

        # Ask Azure OpenAI
        response = openai.ChatCompletion.create(engine=deployment_id, messages=messages, max_tokens=100)

        if 'content' in response['choices'][0]['message']:
            text = response['choices'][0]['message']['content'].replace('\n', ' ').replace(' .', '.').strip()
            print('Azure OpenAI response:' + text)

            comm.send(text, dest=2, tag=sentance)

            sentance += 1
        else:
            print(response['choices'][0]['finish_reason'])


# Continuously listens for speech input to recognize and send as text to Azure OpenAI
def speech_recog(comm):
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    # Should be the locale for the speaker's language.
    speech_config.speech_recognition_language="en-US"

    # 300 ms for BBC news, it should be modefied from person to person, not suitable for conversations of multiple person.
    speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "200")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Azure OpenAI is listening. Say 'Stop' or press Ctrl-Z to end the conversation.")

    done = False

    def send_reg_text(evt):
        if len(evt.result.text) > 1:
            comm.send(evt.result.text, dest=1)


    def stop_cb(evt):
        print('CLOSING on {}'.format(evt))
        speech_recognizer.stop_continuous_recognition_async()
        nonlocal done
        done = True

    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(send_reg_text)
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    speech_recognizer.start_continuous_recognition_async()

    while not done:
        time.sleep(0.5)

def speech_synthesis(comm):
    audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    
    # The language of the voice that responds on behalf of Azure OpenAI.
    speech_config.speech_synthesis_voice_name='zh-CN-XiaoxiaoNeural'
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

    sentance = 0
    while True:

        text = comm.recv(source=1, tag=sentance)
        sentance += 1

        # Azure text to speech output
        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

        # Check result
        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized to speaker for text [{}]".format(text))
        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

# Main
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print('process 0')
        speech_recog(comm)

    elif rank == 1:
        print('process 1')
        ask_openai(comm)

    elif rank == 2:
        print('process 2')
        speech_synthesis(comm)
