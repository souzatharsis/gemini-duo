"""Google Cloud Text-to-Speech provider implementation."""

from google.cloud import texttospeech_v1beta1
from typing import List
import re
import logging
import io
from io import BytesIO
from pydub import AudioSegment
from abc import ABC, abstractmethod
from typing import List, ClassVar, Tuple
from dotenv import load_dotenv
import os
from IPython.display import Audio, display

load_dotenv()

logger = logging.getLogger(__name__)





"""Abstract base class for Text-to-Speech providers."""
class TTSProvider(ABC):
    """Abstract base class that defines the interface for TTS providers."""
    
    # Common SSML tags supported by most providers
    COMMON_SSML_TAGS: ClassVar[List[str]] = [
        'lang', 'p', 'phoneme', 's', 'sub'
    ]
    
    @abstractmethod
    def generate_audio(self, text: str, voice: str, model: str, voice2: str) -> bytes:
        """
        Generate audio from text using the provider's API.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID/name to use
            model: Model ID/name to use
            
        Returns:
            Audio data as bytes
            
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If audio generation fails
        """
        pass

    def get_supported_tags(self) -> List[str]:
        """
        Get set of SSML tags supported by this provider.
        
        Returns:
            Set of supported SSML tag names
        """
        return self.COMMON_SSML_TAGS.copy()
    
    def validate_parameters(self, text: str, voice: str, model: str, voice2: str = None) -> None:
        """
        Validate input parameters before generating audio.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if not text:
            raise ValueError("Text cannot be empty")
        if not voice:
            raise ValueError("Voice must be specified")
        if not model:
            raise ValueError("Model must be specified")
        
    def split_qa(self, input_text: str, ending_message: str, supported_tags: List[str] = None) -> List[Tuple[str, str]]:
        """
        Split the input text into question-answer pairs.

        Args:
            input_text (str): The input text containing Person1 and Person2 dialogues.
            ending_message (str): The ending message to add to the end of the input text.

        Returns:
                List[Tuple[str, str]]: A list of tuples containing (Person1, Person2) dialogues.
        """
        input_text = self.clean_tss_markup(input_text, supported_tags=supported_tags)
        
        # Add placeholder if input_text starts with <Person2>
        if input_text.strip().startswith("<Person2>"):
            input_text = "<Person1> Humm... </Person1>" + input_text

        # Add ending message to the end of input_text
        if input_text.strip().endswith("</Person1>"):
            input_text += f"<Person2>{ending_message}</Person2>"

        # Regular expression pattern to match Person1 and Person2 dialogues
        pattern = r"<Person1>(.*?)</Person1>\s*<Person2>(.*?)</Person2>"

        # Find all matches in the input text
        matches = re.findall(pattern, input_text, re.DOTALL)

        # Process the matches to remove extra whitespace and newlines
        processed_matches = [
            (" ".join(person1.split()).strip(), " ".join(person2.split()).strip())
            for person1, person2 in matches
        ]
        return processed_matches

    def clean_tss_markup(self, input_text: str, additional_tags: List[str] = ["Person1", "Person2"], supported_tags: List[str] = None) -> str:
        """
        Remove unsupported TSS markup tags from the input text while preserving supported SSML tags.

        Args:
            input_text (str): The input text containing TSS markup tags.
            additional_tags (List[str]): Optional list of additional tags to preserve. Defaults to ["Person1", "Person2"].
            supported_tags (List[str]): Optional list of supported tags. If None, use COMMON_SSML_TAGS.
        Returns:
            str: Cleaned text with unsupported TSS markup tags removed.
        """
        if supported_tags is None:
            supported_tags = self.COMMON_SSML_TAGS.copy()

        # Append additional tags to the supported tags list
        supported_tags.extend(additional_tags)

        # Create a pattern that matches any tag not in the supported list
        pattern = r'</?(?!(?:' + '|'.join(supported_tags) + r')\b)[^>]+>'

        # Remove unsupported tags
        cleaned_text = re.sub(pattern, '', input_text)

        # Remove any leftover empty lines
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)

        # Remove asterisks around words
        cleaned_text = re.sub(r'\*(\w+)\*', r'\1', cleaned_text)

        # Ensure closing tags for additional tags are preserved
        for tag in additional_tags:
            cleaned_text = re.sub(f'<{tag}>(.*?)(?=<(?:{"|".join(additional_tags)})>|$)', 
                                f'<{tag}>\\1</{tag}>', 
                                cleaned_text, 
                                flags=re.DOTALL)

        return cleaned_text.strip()



class GeminiMultiTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider with multi-speaker support."""
    
    def __init__(self, api_key: str = None, model: str = "en-US-Studio-MultiSpeaker"):
        """
        Initialize Google Cloud TTS provider.
        
        Args:
            api_key (str): Google Cloud API key
        """
        self.model = model
        try:
            self.client = texttospeech_v1beta1.TextToSpeechClient(
                client_options={'api_key': api_key} if api_key else None
            )
            logger.info("Successfully initialized GeminiMultiTTS client")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiMultiTTS client: {str(e)}")
            raise
            
    def chunk_text(self, text: str, max_bytes: int = 1300) -> List[str]:
        """
        Split text into chunks that fit within Google TTS byte limit while preserving speaker tags.
        
        Args:
            text (str): Input text with Person1/Person2 tags
            max_bytes (int): Maximum bytes per chunk
            
        Returns:
            List[str]: List of text chunks with proper speaker tags preserved
        """
        logger.debug(f"Starting chunk_text with text length: {len(text)} bytes")
        
        # Split text into tagged sections, preserving both Person1 and Person2 tags
        pattern = r'(<Person[12]>.*?</Person[12]>)'
        sections = re.split(pattern, text, flags=re.DOTALL)
        sections = [s.strip() for s in sections if s.strip()]
        logger.debug(f"Split text into {len(sections)} sections")
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            # Extract speaker tag and content if this is a tagged section
            tag_match = re.match(r'<(Person[12])>(.*?)</Person[12]>', section, flags=re.DOTALL)
            
            if tag_match:
                speaker_tag = tag_match.group(1)  # Will be either Person1 or Person2
                content = tag_match.group(2).strip()
                
                # Test if adding this entire section would exceed limit
                test_chunk = current_chunk
                if current_chunk:
                    test_chunk += f"<{speaker_tag}>{content}</{speaker_tag}>"
                else:
                    test_chunk = f"<{speaker_tag}>{content}</{speaker_tag}>"
                    
                if len(test_chunk.encode('utf-8')) > max_bytes and current_chunk:
                    # Store current chunk and start new one
                    chunks.append(current_chunk)
                    current_chunk = f"<{speaker_tag}>{content}</{speaker_tag}>"
                else:
                    # Add to current chunk
                    current_chunk = test_chunk
        
        # Add final chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
            
        logger.info(f"Created {len(chunks)} chunks from input text")
        return chunks

    def split_turn_text(self, text: str, max_chars: int = 500) -> List[str]:
        """
        Split turn text into smaller chunks at sentence boundaries.
        
        Args:
            text (str): Text content of a single turn
            max_chars (int): Maximum characters per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        #print(f"### TEXT: {text}" )
        #print(f"### LENGTH: {len(text)}")
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        sentences = re.split(r'([.!?]+(?:\s+|$))', text)
        sentences = [s for s in sentences if s]
        
        current_chunk = ""
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            separator = sentences[i + 1] if i + 1 < len(sentences) else ""
            complete_sentence = sentence + separator
            
            if len(current_chunk) + len(complete_sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = complete_sentence
                else:
                    # If a single sentence is too long, split at word boundaries
                    words = complete_sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > max_chars:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    current_chunk = temp_chunk
            else:
                current_chunk += complete_sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def merge_audio(self, audio_chunks: List[bytes]) -> bytes:
        """
        Merge multiple MP3 audio chunks into a single audio file.
        
        Args:
            audio_chunks (List[bytes]): List of MP3 audio data
            
        Returns:
            bytes: Combined MP3 audio data
        """
        if not audio_chunks:
            return b""
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        try:
            # Initialize combined audio with first chunk
            combined = None
            valid_chunks = []
            
            for i, chunk in enumerate(audio_chunks):
                try:
                    # Ensure chunk is not empty
                    if not chunk or len(chunk) == 0:
                        logger.warning(f"Skipping empty chunk {i}")
                        continue
                    
                    # Save chunk to temporary file for ffmpeg to process
                    temp_file = f"temp_chunk_{i}.mp3"
                    with open(temp_file, "wb") as f:
                        f.write(chunk)
                    
                    # Create audio segment from temp file
                    try:
                        segment = AudioSegment.from_file(temp_file, format="mp3")
                        if len(segment) > 0:
                            valid_chunks.append(segment)
                            logger.debug(f"Successfully processed chunk {i}")
                        else:
                            logger.warning(f"Zero-length segment in chunk {i}")
                    except Exception as e:
                        logger.error(f"Error processing chunk {i}: {str(e)}")
                    
                    # Clean up temp file
                    import os
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"Error handling chunk {i}: {str(e)}")
                    continue
            
            if not valid_chunks:
                raise RuntimeError("No valid audio chunks to merge")
            
            # Merge valid chunks
            combined = valid_chunks[0]
            for segment in valid_chunks[1:]:
                combined = combined + segment
            
            # Export with specific parameters
            output = BytesIO()
            combined.export(
                output,
                format="mp3",
                codec="libmp3lame",
                bitrate="320k"
            )
            
            result = output.getvalue()
            if len(result) == 0:
                raise RuntimeError("Export produced empty output")
            
            return result
            
        except Exception as e:
            logger.error(f"Audio merge failed: {str(e)}", exc_info=True)
            # If merging fails, return the first valid chunk as fallback
            if audio_chunks:
                return audio_chunks[0]
            raise RuntimeError(f"Failed to merge audio chunks and no valid fallback found: {str(e)}")

    def generate_audio(self, text: str, voice: str = "R", model: str = "en-US-Studio-MultiSpeaker", 
                       voice2: str = "S", ending_message: str = ""):
        """
        Generate audio using Google Cloud TTS API with multi-speaker support.
        Handles text longer than 5000 bytes by chunking and merging.
        """
        logger.info(f"Starting audio generation for text of length: {len(text)}")
        logger.debug(f"Parameters: voice={voice}, voice2={voice2}, model={model}")
        #print("######################### TEXT #########################")
        #print(text)
        #print("######################### END TEXT #########################")
        try:
            # Split text into chunks if needed
            text_chunks = self.chunk_text(text)
            logger.info(f"#########################33 Text split into {len(text_chunks)} chunks")
            audio_chunks = []
            #print(text_chunks[0])
            
            # Process each chunk
            for i, chunk in enumerate(text_chunks, 1):
                logger.debug(f"Processing chunk {i}/{len(text_chunks)}")
                # Create multi-speaker markup
                multi_speaker_markup = texttospeech_v1beta1.MultiSpeakerMarkup()
                #print("######################### CHUNK #########################")
                #print(chunk)
                # Get Q&A pairs for this chunk
                qa_pairs = self.split_qa(chunk, "", self.get_supported_tags())
                logger.debug(f"Found {len(qa_pairs)} Q&A pairs in chunk {i}")
                #print("######################### QA PAIRS #########################")
                #print(qa_pairs)
                # Add turns for each Q&A pair
                for j, (question, answer) in enumerate(qa_pairs, 1):
                    logger.debug(f"Processing Q&A pair {j}/{len(qa_pairs)}")
                    
                    # Split question into smaller chunks if needed
                    question_chunks = self.split_turn_text(question.strip())
                    logger.debug(f"Question split into {len(question_chunks)} chunks")
                    logger.debug(f"######################### Question chunks: {question_chunks}")
                    for q_chunk in question_chunks:
                        logger.debug(f"Adding question turn: '{q_chunk[:50]}...' (length: {len(q_chunk)})")
                        q_turn = texttospeech_v1beta1.MultiSpeakerMarkup.Turn()
                        q_turn.text = q_chunk
                        q_turn.speaker = voice
                        multi_speaker_markup.turns.append(q_turn)
                    
                    # Split answer into smaller chunks if needed
                    if answer:
                        answer_chunks = self.split_turn_text(answer.strip())
                        logger.debug(f"Answer split into {len(answer_chunks)} chunks")
                        logger.debug(f"######################### Answer chunks: {answer_chunks}")
                        for a_chunk in answer_chunks:
                            logger.debug(f"Adding answer turn: '{a_chunk[:50]}...' (length: {len(a_chunk)})")
                            a_turn = texttospeech_v1beta1.MultiSpeakerMarkup.Turn()
                            a_turn.text = a_chunk
                            a_turn.speaker = voice2
                            multi_speaker_markup.turns.append(a_turn)
                
                logger.debug(f"Created markup with {len(multi_speaker_markup.turns)} turns")
                
                # Create synthesis input with multi-speaker markup
                synthesis_input = texttospeech_v1beta1.SynthesisInput(
                    multi_speaker_markup=multi_speaker_markup
                )
                
                logger.debug("Calling synthesize_speech API")
                # Set voice parameters
                voice_params = texttospeech_v1beta1.VoiceSelectionParams(
                    language_code="en-US",
                    name=model
                )
                
                # Set audio config
                audio_config = texttospeech_v1beta1.AudioConfig(
                    audio_encoding=texttospeech_v1beta1.AudioEncoding.MP3,
                    #sample_rate_hertz=44100,  # Specify sample rate
                    #effects_profile_id=['headphone-class-device'],  # Optimize for headphones
                    #speaking_rate=1.0,  # Normal speaking rate
                )
                
                # Generate speech for this chunk
                response = self.client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice_params,
                    audio_config=audio_config
                )

                audio_chunks.append(response.audio_content)
            #print(f"#### Audio chunks: {audio_chunks}")
            #print(f"#### Audio chunks length: {len(audio_chunks)}")
            return audio_chunks
        
            
        except Exception as e:
            logger.error(f"Failed to generate audio: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate audio: {str(e)}") from e
    
    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        # Add any Google-specific SSML tags to the common ones
        return self.COMMON_SSML_TAGS
        
    def validate_parameters(self, text: str, voice: str, model: str) -> None:
        """
        Validate input parameters before generating audio.
        
        Args:
            text (str): Input text
            voice (str): Voice ID
            model (str): Model name
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().validate_parameters(text, voice, model)
        
        # Additional validation for multi-speaker model
        if model != "en-US-Studio-MultiSpeaker":
            raise ValueError(
                "Google Multi-speaker TTS requires model='en-US-Studio-MultiSpeaker'"
            )
        

class TTS:
    def __init__(self, api_key: str, voice: str = "S", model: str = "en-US-Studio-MultiSpeaker", voice2: str = "R", ending_message: str = "Bye Bye"):
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.voice2 = voice2
        self.ending_message = ending_message
        self.provider = GeminiMultiTTS(api_key=self.api_key)

    def generate_audio(self, input_text: str) -> List[bytes]:
        """
        Generate audio data from input text using the TTS provider.
        
        Args:
            input_text (str): The text to convert to audio.
        
        Returns:
            List[bytes]: A list of audio data chunks.
        """
        audio_data_list = self.provider.generate_audio(
            input_text,
            voice=self.voice,
            model=self.model,
            voice2=self.voice2,
            ending_message=self.ending_message,
        )
        return audio_data_list

    def combine_audio(self, audio_data_list: List[bytes]) -> AudioSegment:
        """
        Combine multiple audio data chunks into a single audio segment.
        
        Args:
            audio_data_list (List[bytes]): A list of audio data chunks.
        
        Returns:
            AudioSegment: The combined audio segment.
        """
        combined = AudioSegment.empty()
        for chunk in audio_data_list:
            segment = AudioSegment.from_file(io.BytesIO(chunk))
            combined += segment
        return combined

    def export_audio(self, combined: AudioSegment) -> io.BytesIO:
        """
        Export the combined audio segment to a buffer.
        
        Args:
            combined (AudioSegment): The combined audio segment.
        
        Returns:
            io.BytesIO: The buffer containing the exported audio.
        """
        buffer = io.BytesIO()
        combined.export(
            buffer,
            format="mp3",
            codec="libmp3lame",
            bitrate="320k"
        )
        buffer.seek(0)  # Reset buffer position to start
        return buffer

    def display_audio(self, buffer: io.BytesIO) -> None:
        """
        Display the audio using IPython's Audio display.
        
        Args:
            buffer (io.BytesIO): The buffer containing the audio data.
        """
        display(Audio(buffer.read(), rate=44100))

    def process_text_to_audio(self, input_text: str) -> AudioSegment:
        """
        Process input text to generate, combine, and display audio.
        
        Args:
            input_text (str): The text to convert to audio.
        
        Returns:
            AudioSegment: The combined audio segment.
        """
        audio_data_list = self.generate_audio(input_text)
        combined_audio = self.combine_audio(audio_data_list)
        audio_buffer = self.export_audio(combined_audio)
        self.display_audio(audio_buffer)
        return combined_audio


