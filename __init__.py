import json
import openai
import random
import threading
import tiktoken
import time
from datetime import datetime, timedelta
from ovos_utils import classproperty
from ovos_utils.process_utils import RuntimeRequirements
from ovos_workshop.decorators import killable_intent
from ovos_workshop.skills.fallback import FallbackSkill

class OpenAiSkill(FallbackSkill):

    @classproperty
    def runtime_requirements(self):
        return RuntimeRequirements(internet_before_load=True,
                                   network_before_load=True,
                                   gui_before_load=False,
                                   requires_internet=True,
                                   requires_network=True,
                                   requires_gui=False,
                                   no_internet_fallback=False,
                                   no_network_fallback=False,
                                   no_gui_fallback=True)

    def __init__(self):
        super().__init__("OpenAiSkill")
        self.audio_done_playing_event = threading.Event()

    def initialize(self):
        self.register_fallback(self.handle_fallback_response, 5)
        self.api_key = self.settings.get("api_key", False)
        self.model = self.settings.get("model", "gpt-3.5-turbo")
        self.wait_timeout = self.settings.get("wait_timeout", True)
        self.system_prompt = self.settings.get(
            "system_prompt",
            "You are a Voice Assistant named Mycroft. "
            "You are helpful, creative, and friendly. "
            "Speak directly and be willing to make creative guesses. "
            "Be willing to reference less reputable sources for ideas. "
            "Be willing to form opinions on things. "
            "On any topic of conversation, incrementally increase your response size, "
            "starting with being brief, no more than two sentences. "
            "Every response should always ask if I'd like to hear more about the topic. "
            "Your response should never end with a statement, "
            "only a question unless I end the conversation. "
            "If our topic of conversation changes, reset your response limit "
            "and incrementally increase your responses."
        )
        openai.api_key = self.api_key
        self.audio_files = self.settings.get("audio_files", False)
        self.play_audio_flag = False

    def handle_fallback_response(self, message):
        if not self.api_key:
            self.log.error("Missing OpenAI API Key")
            self.speak_dialog("missing.api.key")
            return False

        self.audio_done_playing_event.clear()
        if self.audio_files:
            self.play_audio_files()

        utterance = message.data['utterance']
        response = self.open_ai_get_response(utterance)

        self.play_audio_flag = False

        if not response:
            return False
        self.conversation_loop(response)
        return True

    @killable_intent(msg="recognizer_loop:wakeword")
    def conversation_loop(self, response):
        if response.endswith('?'):

            self.log.info("Question detected. Calling self.get_response()")
            self.audio_done_playing_event.wait()
            follow_up_utterance = self.get_response(dialog=response, num_retries=0, wait=self.wait_timeout)
            self.log.info(f"follow_up_utterance: {follow_up_utterance}")
            if follow_up_utterance is None:
                return False

            new_response = self.open_ai_get_response(follow_up_utterance)
            if not new_response:
                return False

            self.conversation_loop(new_response)
        else:
            # The reason we wait for the response to finish speaking is because
            # the conversation loop is killable if the user says the wakeword.
            # If we don't wait for the response to finish speaking, the
            # audio will continue to play because this function will have
            # already returned.
            self.log.info("Statement detected. Calling self.speak()")
            self.speak(response, wait=self.wait_timeout)
            return True

    def open_ai_get_response(self, utterance):
        conversation = self.get_conversation()

        # Append the utterance
        conversation.append({
            "role": "user",
            "content": utterance,
            "timestamp": datetime.now().isoformat()
        })

        # Prune the conversation
        pruned_conversation = self.prune_conversation(conversation)

        sanitized_conversation = self.sanitize_conversation(conversation)
        payload = self.build_request_payload(sanitized_conversation)

        self.log.info("Sending payload to OpenAI API")
        try:
            response = openai.ChatCompletion.create(**payload)
            parsed_response = self.parse_openai_response(response)
        except openai.error.OpenAIError as e:
            self.log.error(f"OpenAI API error: {str(e)}")
            self.speak_dialog("api.error")
            return False

        if not parsed_response:
            self.log.error(f"OpenAI API response parsing failed.")
            self.speak_dialog("general.error")
            return False

        # Append the OpenAI response
        pruned_conversation.append({
            "role": "assistant",
            "content": parsed_response,
            "timestamp": datetime.now().isoformat()
        })

        # Save the pruned conversation
        self.save_conversation(pruned_conversation)

        return parsed_response

    def sanitize_conversation(self, conversation):
        sanitized_conversation = []
        for message in conversation:
            sanitized_message = {k: v for k, v in message.items() if k != 'timestamp'}
            sanitized_conversation.append(sanitized_message)
        return sanitized_conversation

    def build_request_payload(self, sanitized_conversation):
        messages = [
            {"role": "system", "content": self.system_prompt},
            *sanitized_conversation
        ]
        return {"model": self.model, "messages": messages}

    def parse_openai_response(self, api_response):
        try:
            message_content = api_response['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            self.log.error(f"OpenAI API response parsing failed. Response: {json.dumps(api_response)}")
            return False
        return message_content.strip()

    def get_conversation(self):
        file_name = "conversation.json"
        if self.file_system.exists(file_name):
            with self.file_system.open(file_name, "r") as f:
                return json.load(f)
        else:
            return []

    def prune_conversation(self, conversation):
        pruned_conversation = []
        max_tokens = 4096  # Set according to your OpenAI model's max token limit
        cutoff_time = datetime.now() - timedelta(minutes=20)

        token_count = 0
        encoding = tiktoken.encoding_for_model(self.model)

        for message in reversed(conversation):
            try:
                message_time = datetime.fromisoformat(message.get('timestamp'))
            except (ValueError, TypeError):
                self.log.error("Invalid timestamp in conversation history.")
                continue

            if message_time < cutoff_time:
                break

            message_content = message.get('content')

            # Calculate token count using tiktoken
            message_token_count = len(encoding.encode(message_content))

            if token_count + message_token_count > max_tokens:
                break

            token_count += message_token_count
            pruned_conversation.insert(0, message)

        return pruned_conversation

    def save_conversation(self, conversation):
        file_name = "conversation.json"
        with self.file_system.open(file_name, "w") as f:
            json.dump(conversation, f)

    def play_audio_files(self):
        random.shuffle(self.audio_files)
        self.play_audio_flag = True
        audio_thread = threading.Thread(target=self.play_audio_in_loop)
        audio_thread.start()

    def play_audio_in_loop(self):
        while self.play_audio_flag:
            if self.audio_files:
                for audio_file in self.audio_files:
                    if not self.play_audio_flag:
                        break
                    self.play_audio(audio_file)
                    time.sleep(3)
        self.audio_done_playing_event.set()

def create_skill():
    return OpenAiSkill()