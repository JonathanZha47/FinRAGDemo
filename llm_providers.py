from typing import Optional, Dict, Any
import os
from abc import ABC, abstractmethod
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError, OpenAIError
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import tiktoken

load_dotenv()

def truncate_prompt(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within token limit."""
    try:
        if model.startswith("gpt"):
            encoding = tiktoken.encoding_for_model(model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default for other models
            
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        # Leave some room for the response and system message
        truncated_tokens = tokens[:max_tokens - 1000]
        return encoding.decode(truncated_tokens)
    except Exception as e:
        print(f"Error in truncation: {str(e)}")
        # Fallback to simple character-based truncation
        char_per_token = 4  # Rough estimate
        safe_length = (max_tokens - 1000) * char_per_token
        return text[:safe_length]

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        try:
            if not api_key.startswith('sk-'):
                raise ValueError("Invalid API key format. OpenAI API keys should start with 'sk-'")
                
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.max_tokens = 14000 if "gpt-4" in model else 14000  # Conservative limits
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            # Truncate prompt if needed
            truncated_prompt = truncate_prompt(prompt, self.max_tokens, self.model)
            
            # Prepare messages with system prompt if provided
            messages = [{"role": "user", "content": truncated_prompt}]
            if kwargs.get('system_prompt'):
                messages.insert(0, {"role": "system", "content": kwargs['system_prompt']})
                
            # Create chat completion with minimal parameters to avoid errors
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_output_tokens', 500)  # Use the max_output_tokens from kwargs
            )
            return response.choices[0].message.content
            
        except AuthenticationError as e:
            return f"OpenAI Error: Authentication failed. Please check your API key. Details: {str(e)}"
        except RateLimitError as e:
            return f"OpenAI Error: Rate limit exceeded. Please try again later. Details: {str(e)}"
        except APIConnectionError as e:
            return f"OpenAI Error: Connection failed. Please check your internet connection. Details: {str(e)}"
        except APIError as e:
            return f"OpenAI Error: API error occurred. Details: {str(e)}"
        except OpenAIError as e:
            return f"OpenAI Error: An error occurred. Details: {str(e)}"
        except Exception as e:
            return f"OpenAI Error: Unexpected error occurred. Details: {str(e)}"

class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        try:
            self.client = InferenceClient(token=api_key)
            self.model = model
            self.max_tokens = 24000  # Conservative limit for Mixtral
        except Exception as e:
            print(f"Error initializing HuggingFace client: {str(e)}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            # Truncate prompt if needed
            truncated_prompt = truncate_prompt(prompt, self.max_tokens)
            
            # Format prompt for instruction-tuned models
            formatted_prompt = f"""<s>[INST] {truncated_prompt} [/INST]"""
            
            response = self.client.text_generation(
                formatted_prompt,
                model=self.model,
                temperature=kwargs.get('temperature', 0.1),
                max_new_tokens=kwargs.get('max_output_tokens', 512),  # Use the max_output_tokens from kwargs
                repetition_penalty=1.1
            )
            return response
        except Exception as e:
            return f"HuggingFace Error: {str(e)}"

class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "mistralai/mixtral-8x7b-instruct"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        # Set max tokens based on model
        self.max_tokens = 24000 if "mixtral" in model else 8000  # Conservative limits

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            # Truncate prompt if needed
            truncated_prompt = truncate_prompt(prompt, self.max_tokens)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Financial Advisor Bot",
                "Content-Type": "application/json",
                "OpenRouter-Referrer": "http://localhost:8501",
            }
            
            messages = [{"role": "user", "content": truncated_prompt}]
            if kwargs.get('system_prompt'):
                messages.insert(0, {"role": "system", "content": kwargs['system_prompt']})
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get('temperature', 0.1),
                "max_tokens": kwargs.get('max_output_tokens', 1000),  # Use the max_output_tokens from kwargs
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                response_json = response.json()
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    if 'message' in response_json['choices'][0]:
                        return response_json['choices'][0]['message']['content']
                    elif 'text' in response_json['choices'][0]:
                        return response_json['choices'][0]['text']
                return "Error: Unexpected response format from OpenRouter"
            else:
                error_message = response.json().get('error', {}).get('message', 'Unknown error')
                if "maximum context length" in error_message.lower():
                    return f"OpenRouter Error: Context too long. The response has been truncated. Details: {error_message}"
                return f"OpenRouter Error: {error_message}"
                
        except requests.exceptions.RequestException as e:
            return f"OpenRouter Error: {str(e)}"
        except Exception as e:
            return f"OpenRouter Error: {str(e)}"

def get_llm_provider(provider_name: str) -> Optional[BaseLLMProvider]:
    providers = {
        'openai': (OpenAIProvider, 'OPENAI_API_KEY', 'gpt-3.5-turbo'),
        'huggingface': (HuggingFaceProvider, 'HUGGINGFACE_API_KEY', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
        'openrouter': (OpenRouterProvider, 'OPENROUTER_API_KEY', 'mistralai/mixtral-8x7b-instruct')
    }

    if provider_name not in providers:
        return None

    ProviderClass, api_key_name, default_model = providers[provider_name]
    api_key = os.getenv(api_key_name)
    
    if not api_key:
        raise ValueError(f"Please set {api_key_name} in your .env file")
    
    try:
        return ProviderClass(api_key, default_model)
    except Exception as e:
        print(f"Error initializing {provider_name} provider: {str(e)}")
        return None 