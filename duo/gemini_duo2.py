from typing import List, Dict, Tuple
from gemini_duo_content import WebsiteExtractor
from gemini_duo_tts import TTS

import os
import google.generativeai as genai
from google.generativeai import caching
import datetime
import time
from langchain import hub

from dotenv import load_dotenv
import os
from enum import Enum
from pydub import AudioSegment


PROMPT_TEMPLATE = "souzatharsis/duo"
PROMPT_COMMIT = "c0ecab9a" #"75e56f77" 
PROMPT_QUIZ_TEMPLATE = "souzatharsis/duo_quiz"
PROMPT_QUIZ_COMMIT = "9e9a7500"

GEMINI_MODEL_NAME = "gemini-1.5-pro-002"
GEMINI_API_KEY_LABEL = "GEMINI_API_KEY"
GEMINI_GROUNDED_MODEL_NAME = "gemini-1.5-flash-002"

# Load environment variables from .env file
load_dotenv()

class KnowledgeBaseStatus(Enum):
    NOT_LOADED = "not loaded"
    JUST_LOADED = "just loaded"
    LOADED = "loaded"

import typing_extensions as typing
class Round(typing.TypedDict):
    person1: str
    person2: str

class LLMBackend:
    """
    A backend class for managing LLM interactions.
    """
    CACHE_TTL = 60 # cache time-to-live in minutes
    def __init__(
        self,
        model_name: str = GEMINI_MODEL_NAME,
        api_key_label: str = GEMINI_API_KEY_LABEL,
        conversation_config: Dict = {},
        input: str = "",
        cache_ttl: int = CACHE_TTL,
    ):
        """
        Initialize the LLMBackend.

        Args:
                temperature (float): The temperature for text generation.
                model_name (str): The name of the model to use.
        """
        self.model_name = model_name
        
        #secret_value = UserSecretsClient().get_secret(api_key_label)#TODO
        genai.configure(api_key=os.environ[api_key_label])
        

        self.cache = caching.CachedContent.create(
            model=model_name,
            display_name='due_knowledge_base', # used to identify the cache
            system_instruction=(
                self.compose_prompt(input, conversation_config)
            ),
            ttl=datetime.timedelta(minutes=cache_ttl),
        )

        
        
        self.model = genai.GenerativeModel.from_cached_content(cached_content=self.cache)
    
    def compose_prompt(self, input:str, conversation_config: Dict) -> str:
        """
        Compose a prompt for the Gemini Duo model using a LangChain prompt template.
        """
        prompt_template = hub.pull(f"{PROMPT_TEMPLATE}:{PROMPT_COMMIT}")
        prompt = prompt_template.invoke({"memory": input,
          "output_language": conversation_config["output_language"],
          "conversation_style": conversation_config["conversation_style"],
          "roles_person1": conversation_config["roles_person1"],
          "roles_person2": conversation_config["roles_person2"],
          "dialogue_structure": conversation_config["dialogue_structure"],
          "engagement_techniques": conversation_config["engagement_techniques"],
          "input_texts": ""})
        
        return prompt.messages[0].content


class Quiz:
    """
    A backend class for managing quiz generation.
    """
    CACHE_TTL = 60 # cache time-to-live in minutes
    def __init__(
        self,
        model_name: str = GEMINI_MODEL_NAME,
        api_key_label: str = GEMINI_API_KEY_LABEL,
        input: str = "",
        cache_ttl: int = CACHE_TTL,
        add_citations: bool = False,
        num_questions: int = 10,
    ):
        """
        Initialize the Quiz.

        Args:

        """
        self.model_name = model_name
        self.input = input
        #secret_value = UserSecretsClient().get_secret(api_key_label)#TODO
        genai.configure(api_key=os.environ[api_key_label])

        self.cache = caching.CachedContent.create(
            model=model_name,
            display_name='due_quiz', # used to identify the cache
            system_instruction=(
                self.compose_prompt(num_questions, add_citations)),
                contents=input,
            ttl=datetime.timedelta(minutes=cache_ttl),
        )

        
        
        self.model = genai.GenerativeModel.from_cached_content(cached_content=self.cache)

    def compose_prompt(self, num_questions: int = 10, add_citations: bool = True) -> str:
        if add_citations:
            citations = "In the Quiz answers, at the end, make sure to add Input ID indicating referenced content pertaining to the answer to the question."
        else:
            citations = ""
        prompt_template = hub.pull(f"{PROMPT_QUIZ_TEMPLATE}:{PROMPT_QUIZ_COMMIT}")
        prompt = prompt_template.invoke({"num_questions": num_questions,
          "citations": citations,
          "input_texts": ""})
        return prompt.messages[0].content

    
    def generate(self, msg:str="") -> str:
        msg = f"Generate: {msg}"
        response = self.model.generate_content([msg])
        return response


class ContentGenerator:
    """
    A class to handle content generation using the Gemini model.
    """
    def __init__(self, model):
        """
        Initialize the ContentGenerator with a model and input content.
        
        Args:
            model: The Gemini model instance to use for generation
            input_content: The input content to generate from
        """
        self.cached_model = model
        self.non_cached_model = None

    # cached generation
    def generate(self, input_content, user_instructions=""):
        """
        Generate content using the model.
        
        Returns:
            The model's response
        """
        prompt=f"""
        USER_INSTRUCTIONS: Make sure to follow these instructions: {user_instructions}
        INPUT:{input_content}
        """
        llm_config = genai.generation_config={"response_mime_type": "application/json",
                   "response_schema": list[Round]}
        response = self.cached_model.generate_content([(
            prompt)],
            generation_config=llm_config)
        return response
    
    # grounded generation
    def generate_grounded(self, topic: str) -> str:
        """
        Generate content based on a given topic using a generative model.

        Args:
            topic (str): The topic to generate content for.

        Returns:
            str: Generated content based on the topic.
        """

        if self.non_cached_model is None:
            self.non_cached_model = genai.GenerativeModel(f'models/{GEMINI_GROUNDED_MODEL_NAME}')
        model = self.non_cached_model
        topic_prompt = f'Be detailed. Search for {topic}'
        response = model.generate_content(contents=topic_prompt, tools='google_search_retrieval')
        return response


import json

class Client:
    """
    A client class for managing conversational AI interactions using the Gemini model.
    
    This class handles content extraction, generation of conversational content, and audio output
    using the Gemini model. It maintains conversation state and configuration.
    """
    CONVERSATION_CONFIG_DEFAULTS = {
        "conversation_style": [
            "engaging",
            "fast-paced", 
            "enthusiastic"
        ],
        "roles_person1": "Harvard Student: questioner/clarifier",
        "roles_person2": "Harvard Professor",
        "dialogue_structure": [
            "Introduction",
            "Main Content Summary",
            "Conclusion"
        ],
        "output_language": "English",
        "engagement_techniques": [
            "rhetorical questions",
            "analogies",
            "critique"
        ]
    }

    def __init__(self, conversation_config: Dict = CONVERSATION_CONFIG_DEFAULTS, knowledge_base: List[str] = []):
        """
        Initialize the Client class with conversation configuration.

        Args:
            conversation_config (Dict): Configuration dictionary for the conversation.
                                      Defaults to CONVERSATION_CONFIG_DEFAULTS.
        """
        self.knowledge_base = knowledge_base # user-provided foundation knowledge represented as a list of urls
        self.reference_id = 0 # unique ID for each input
        self.input = "" # short-term memory, i.e. current input to be studied
        self.urls = [] # input list of URLs to extract content from
        self.response = "" # latest response from LLM
        self.urls_memory = [] # cumulative list of URLs to extract content from
        self.input_memory = "" # long-term memory, i.e. cumulative input + knowledge base
        self.response_memory = "" # long-term response memory, i.e. cumulative responses
        self.extractor = WebsiteExtractor() # extractor for content from URLs
        self.conversation_config = conversation_config # user-provided conversation configuration
        self.knowledge_base_status = KnowledgeBaseStatus.NOT_LOADED  # Use enum for knowledge base status
        self.quiz_instance = None

        self.add_knowledge_base(self.knowledge_base) 

        self.llm = LLMBackend(conversation_config=self.conversation_config,
                              input=self.input_memory
                              ) # llm with cached content
        
        self.content_generator = ContentGenerator(model=self.llm.model) # content generator with cached llm
        self.tts = TTS(api_key=os.environ["GEMINI_API_KEY"])
    def add_knowledge_base(self, urls: List[str]) -> None:
        """
        Add URLs to the knowledge base.
        """
        self.add(urls)
        self.knowledge_base_status = KnowledgeBaseStatus.JUST_LOADED  # Update status using enum
    

    def refocus(self) -> None:
        """
        Clear the current conversation state while preserving memory.
        
        Resets input, URLs, and image URLs to prepare for new content.
        """
        self.input = ""
        self.urls = []

    def reset(self) -> None:
        """
        Completely reset the conversation state.
        
        Clears all input, memory, and conversation history.
        """
        self.refocus()
        self.urls_memory = []
        self.input_memory = ""
        self.response_memory = ""

    def add(self, urls: List[str], refocus: bool = False) -> None:
        """
        Extract content from URLs and add it to the conversation input.

        Args:
            urls (List[str]): List of URLs to extract content from.
            refocus (bool): Whether to clear current conversation state before adding.
                          Defaults to True.
        """
        # resets input either if user wants to refocus or if KB has just been loaded since it's in cache already
        if refocus or self.knowledge_base_status == KnowledgeBaseStatus.JUST_LOADED: 
            self.refocus()
        
        self.urls = urls

        # Add new content to input following CIC format to enable citations
        for url in urls:
            self.urls_memory.append(url)
            content = self.extractor.extract_content(url)
            formatted_content = f"ID: {self.reference_id} | {content} | END ID: {self.reference_id}"
            self.input += formatted_content + "\n" 
            self.reference_id += 1
        
        # Update memory
        self.input_memory = self.input_memory + self.input

        if self.knowledge_base_status == KnowledgeBaseStatus.JUST_LOADED:
            self.knowledge_base_status = KnowledgeBaseStatus.LOADED

    def research(self, topic: str, transcript_only: bool = False) -> Tuple[str, AudioSegment]:
        """
        Generate research content on a given topic.

        Args:
            topic (str): The topic to research and generate content for.
            transcript_only (bool): Whether to only return the transcript.
                                Defaults to False.

        Returns:
            Tuple[str, AudioSegment]: Generated research content in conversational format.
        """
        topic_content = self.content_generator.generate_grounded(topic)
        self.response = self.content_generator.generate(
            input_content=topic_content
        )
        self.response_memory = self.response_memory + self.response.text
        return (self.response.text, self.tts.process_text_to_audio(self.response.text) if not transcript_only else None)

    def __json_to_text(self, json_response) -> str:
        y = json_response

        combined_dialogue = ""
        for round in y:
            person1_text = round['person1']
            person2_text = round.get('person2', '<Person2>Bye Bye</Person2>')
            
            if not person1_text.startswith('<Person1>'):
                person1_text = '<Person1>' + person1_text
            if not person1_text.endswith('</Person1>'):
                person1_text = person1_text + '</Person1>'
                
            if not person2_text.startswith('<Person2>'):
                person2_text = '<Person2>' + person2_text
            if not person2_text.endswith('</Person2>'):
                person2_text = person2_text + '</Person2>'
                
            combined_dialogue += person1_text + person2_text
        
        return combined_dialogue
    
    def qa(self, msg: str = "", add_citations: bool = False, transcript_only: bool = False) -> Tuple[str, AudioSegment]:
        """
        Generate Q&A content based on the current conversation input.

        Args:
            msg (str): Additional instructions for content generation.
            longform (bool): Whether to generate detailed, long-form content.
                           Defaults to False.
            add_citations (bool): Whether to include input citations in responses.
                                Defaults to False.
            transcript_only (bool): Whether to only return the transcript.
                                Defaults to False.

        Returns:
            str: Generated Q&A conversation content.
        """
        if add_citations:
            msg = msg + "\n\n For key statements, add Input ID to the response."

        self.response = self.content_generator.generate(
            input_content=self.input,
            user_instructions=msg
        )
        print(self.response.usage_metadata)
        #

        response_json = json.loads(self.response.text)
        response_text = self.__json_to_text(response_json)
        self.response_memory = self.response_memory + response_text

        return (response_text, self.tts.process_text_to_audio(response_text) if not transcript_only else None, response_json)

    def set_conversation_config(self, **kwargs) -> None:
        """
        Update the conversation configuration settings.

        Args:
            **kwargs: Arbitrary keyword arguments to update in conversation_config.
                        Defaults values in 

        Examples:
            >>> client = Client()
            >>> client.set_conversation_config(
                ...     podcast_name="Tech Talk",
                ...     podcast_tagline="Exploring the Future of Technology",
                ... )
        """
        for key, value in kwargs.items():
            self.conversation_config[key] = value
        
        self.content_generator = ContentGenerator(
            model_name=self.MODEL_NAME,
            api_key_label=self.API_KEY_LABEL,
            conversation_config=self.conversation_config
        )

    def get_transcript(self) -> str:
        """
        Retrieve the full conversation transcript from memory.

        Returns:
            str: Complete transcript of all responses in the conversation.
        """
        return self.response_memory
    
    def recap(self, msg: str = "", add_citations: bool = False) -> Tuple[str, AudioSegment]:
        """
        Generate a recap of the conversation based on response memory.

        Args:
            msg (str): Additional instructions for recap generation.
            longform (bool): Whether to generate detailed recap content.
                           Defaults to False.
            add_citations (bool): Whether to include citations in recap.
                                Defaults to True.

        Returns:
            str: Generated conversation recap.
        """
        self.input = self.response_memory
        return self.qa(msg, add_citations=add_citations)
    
    def quiz(self, msg: str = "", add_citations: bool = True, num_questions: int = 10) -> str:
        """
        Generate a quiz based on full input memory.
        """
        self.quiz_instance = Quiz(
                         input=self.input_memory,
                         add_citations=add_citations,
                         num_questions=num_questions)
        response = self.quiz_instance.generate(msg)
        print(response.usage_metadata)
        return response.text
