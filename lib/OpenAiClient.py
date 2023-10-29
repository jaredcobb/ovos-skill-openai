import requests
import json
import copy
from ovos_utils.log import LOG

class OpenAiClient:
    def __init__(self, api_key, model, system_prompt):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
    
    def chat(self, conversation):
        payload = self._build_request_payload(conversation)
        response = self._make_api_call(payload)
        message_content = self._parse_response(response)
        return message_content

    def _build_request_payload(self, conversation):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
        # Clone the conversation array and remove timestamp fields
        sanitized_conversation = copy.deepcopy(conversation)
        for message in sanitized_conversation:
            if "timestamp" in message:
                del message["timestamp"]
        
        messages.extend(sanitized_conversation)
        
        payload = {
            "model": self.model,
            "messages": messages
        }
        return json.dumps(payload)
    
    def _make_api_call(self, payload):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=payload
        )
        return response.json()

    def _parse_response(self, api_response):
        try:
            message_content = api_response["choices"][0]["message"]["content"]
            LOG.error(f"OpenAI API response successful. Response: {json.dumps(api_response)}")
        except (KeyError, IndexError, TypeError):
            LOG.error(f"OpenAI API response parsing failed. Response: {json.dumps(api_response)}")
            message_content = "OpenAI API response parsing failed. Please check the logs."
        return message_content.strip()