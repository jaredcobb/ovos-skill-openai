import json
import tiktoken
import threading
import random
import time
from ovos_utils import classproperty
from ovos_utils.log import LOG
from ovos_utils.process_utils import RuntimeRequirements
from ovos_utils.sound import play_acknowledge_sound
from ovos_workshop.skills.fallback import FallbackSkill
from ovos_utils.sound import play_audio
from datetime import datetime, timedelta
from .lib.OpenAiClient import OpenAiClient

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

    def initialize(self):
        self.register_fallback(self.handle_fallback_response, 3)

        self.api_key = self.settings.get("api_key", False)
        self.model = self.settings.get("model", "gpt-3.5-turbo")
        self.system_prompt = self.settings.get("system_prompt", "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.")
        self.audio_files = self.settings.get("audio_files", False)
        self.play_audio_flag = False

        self.openai_client = OpenAiClient(self.api_key, self.model, self.system_prompt)

    def handle_fallback_response(self, message):
        if not self.api_key:
            LOG.error("OpenAI Skill: Missing OpenAI API Key")
            self.speak("The OpenAI API key is missing. Please check your configuration.")
            return False

        if self.audio_files:
            random.shuffle(self.audio_files)
            self.play_audio_flag = True
            audio_thread = threading.Thread(target=self.play_audio_in_loop)
            audio_thread.start()

        utterance = message.data['utterance']
        response = self.open_ai_get_response(utterance)

        self.play_audio_flag = False

        if not response:
            return False
        self.speak(response)
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
        
        response = self.openai_client.chat(pruned_conversation)
        
        # Append the OpenAI response
        pruned_conversation.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        # Save the pruned conversation
        self.save_conversation(pruned_conversation)
        
        return response

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
                LOG.error("Invalid timestamp in conversation history.")
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

    def play_audio_in_loop(self):
        while self.play_audio_flag:
            if self.audio_files:
                for audio_file in self.audio_files:
                    if not self.play_audio_flag:
                        break
                    process = play_audio(audio_file)
                    if process:
                        process.wait()
                    time.sleep(0.5)

def create_skill():
    return OpenAiSkill()