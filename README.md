# OVOS OpenAI Skill

## Description

The `ovos-skill-openai` project is an OpenVoiceOS fallback skill designed to forward user requests to the OpenAI GPT models. It maintains conversational context using the OpenAI Conversations API. This allows for a more coherent and engaging user experience.

## Features

- Fallback to GPT models for queries
- Context management through Conversations API
- Customizable through `settings.json`

## Prerequisites

- OpenVoiceOS installed
- Python 3.x

## Installation

To install the `ovos-skill-openai` skill, run the following command:

```bash
pip3 install git+https://github.com/jaredcobb/ovos-skill-openai.git
```

## Configuration

The `settings.json` configuration file may contain the following properties. Only `api_key` is required:

- `api_key`: Your OpenAI API key [REQUIRED]
- `model`: The OpenAI GPT model you wish to use (default is "gpt-3.5-turbo")
- `system_prompt`: The system prompt for initiating conversations with the GPT model. This is how you give the model some personality. We have a basic default.
- `wait_timeout`: The maximum amount of time OVOS should wait while speaking dialog to you. Some ChatGPT responses may take a while to finish.
- `audio_files`: An array of audio files (one or many) to shuffle and play while we wait for the OpenAI response. The `gpt-4` model is particularly slow at times. This gives the user some feedback that it's processing. Must be an absolute path to a file on the device. Audio files should be less than three seconds long.

### Sample JSON Config

```json
{
  "api_key": "your-api-key-here",
  "model": "gpt-3.5-turbo",
  "system_prompt": "You are a helpful voice assistant with a friendly tone and fun sense of humor",
  "wait_timeout": 90,
  "audio_files": [
    "/home/ovos/.config/files/audio_files_example_01.mp3",
    "/home/ovos/.config/files/audio_files_example_02.mp3",
    "/home/ovos/.config/files/audio_files_example_03.mp3",
    "/home/ovos/.config/files/audio_files_example_04.mp3"
  ]
}
```

## Usage

Once installed and configured, the skill will automatically forward queries it receives as fallbacks to the configured GPT model.

## License

This project is licensed under the Apache License. See the LICENSE file for details.

## Contributing

For contributing guidelines, please refer to the `CONTRIBUTING.md` file.

## Support

If you encounter any issues, please open an issue on GitHub.
