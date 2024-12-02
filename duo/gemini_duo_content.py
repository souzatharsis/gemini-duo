"""
Content Extractor Module

This module provides functionality to extract content from various sources including
websites, and local PDF files.
"""






"""
Website Extractor sub-module

This module is responsible for extracting clean text content from websites using
BeautifulSoup for local HTML parsing instead of the Jina AI API.
"""
import requests
import re
import html
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import List

import logging
logger = logging.getLogger(__name__)

WEBSITE_EXTRACTOR = {
    "markdown_cleaning": {
        "remove_patterns": [
            r'\[.*?\]',      # Remove square brackets and their contents
            r'\(.*?\)',      # Remove parentheses and their contents
            r'^\s*[-*]\s',   # Remove list item markers
            r'^\s*\d+\.\s',  # Remove numbered list markers
            r'^\s*#+'        # Remove markdown headers
        ]
    },
    "unwanted_tags": [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript"
    ],
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "timeout": 10  # Request timeout in seconds
}

class WebsiteExtractor:
	def __init__(self):
		"""
		Initialize the WebsiteExtractor.
		"""
		self.website_extractor_config = WEBSITE_EXTRACTOR
		self.unwanted_tags = self.website_extractor_config['unwanted_tags']
		self.user_agent = self.website_extractor_config['user_agent']
		self.timeout = self.website_extractor_config['timeout']
		self.remove_patterns = self.website_extractor_config['markdown_cleaning']['remove_patterns']

	def extract_content(self, url: str) -> str:
		"""
		Extract clean text content from a website using BeautifulSoup.

		Args:
			url (str): Website URL.

		Returns:
			str: Extracted clean text content.

		Raises:
			Exception: If there's an error in extracting the content.
		"""
		try:
			# Normalize the URL
			normalized_url = self.normalize_url(url)

			# Request the webpage
			headers = {'User-Agent': self.user_agent}
			response = requests.get(normalized_url, headers=headers, timeout=self.timeout)
			response.raise_for_status()  # Raise an exception for bad status codes

			# Parse the page content with BeautifulSoup
			soup = BeautifulSoup(response.text, 'html.parser')

			# Remove unwanted elements
			self.remove_unwanted_elements(soup)

			# Extract and clean the text content
			raw_text = soup.get_text(separator="\n")  # Get all text content
			cleaned_content = self.clean_content(raw_text)

			return cleaned_content
		except requests.RequestException as e:
			logger.error(f"Failed to extract content from {url}: {str(e)}")
			raise Exception(f"Failed to extract content from {url}: {str(e)}")
		except Exception as e:
			logger.error(f"An unexpected error occurred while extracting content from {url}: {str(e)}")
			raise Exception(f"An unexpected error occurred while extracting content from {url}: {str(e)}")

	def normalize_url(self, url: str) -> str:
		"""
		Normalize the given URL by adding scheme if missing and ensuring it's a valid URL.

		Args:
			url (str): The URL to normalize.

		Returns:
			str: The normalized URL.

		Raises:
			ValueError: If the URL is invalid after normalization attempts.
		"""
		# If the URL doesn't start with a scheme, add 'https://'
		if not url.startswith(('http://', 'https://')):
			url = 'https://' + url

		# Parse the URL
		parsed = urlparse(url)

		# Ensure the URL has a valid scheme and netloc
		if not all([parsed.scheme, parsed.netloc]):
			raise ValueError(f"Invalid URL: {url}")

		return parsed.geturl()

	def remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
		"""
		Remove unwanted elements from the BeautifulSoup object.

		Args:
			soup (BeautifulSoup): The BeautifulSoup object to clean.
		"""
		for tag in self.unwanted_tags:
			for element in soup.find_all(tag):
				element.decompose()

	def clean_content(self, content: str) -> str:
		"""
		Clean the extracted content by removing unnecessary whitespace and applying
		custom cleaning patterns.

		Args:
			content (str): The content to clean.

		Returns:
			str: Cleaned text content.
		"""
		# Decode HTML entities
		cleaned_content = html.unescape(content)

		# Remove extra whitespace
		cleaned_content = re.sub(r'\s+', ' ', cleaned_content)

		# Remove extra newlines
		cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

		# Apply custom cleaning patterns from config
		for pattern in self.remove_patterns:
			cleaned_content = re.sub(pattern, '', cleaned_content)

		return cleaned_content.strip()

"""
PDF Extractor Module

This module provides functionality to extract text content from PDF files.
It handles the reading of PDF files, text extraction, and normalization of
the extracted content, including handling of special characters and accents.
"""

import pymupdf
import logging
import os
import unicodedata

class PDFExtractor:
	def extract_content(self, file_path: str) -> str:
		"""
		Extract text content from a PDF file, handling foreign characters and special characters.
		Accents are removed from the text.

		Args:
			file_path (str): Path to the PDF file.

		Returns:
			str: Extracted text content with accents removed and properly handled characters.
		"""
		try:
			doc = pymupdf.open(file_path)
			content = " ".join(page.get_text() for page in doc)
			doc.close()
			
			# Normalize the text to handle special characters and remove accents
			normalized_content = unicodedata.normalize('NFKD', content)

			return normalized_content
		except Exception as e:
			logger.error(f"Error extracting PDF content: {str(e)}")
			raise





"""
Content Extractor Module

This module provides functionality to extract content from various sources including
websites, and PDF files.
It serves as a central hub for content extraction, delegating to specialized extractors based on the source type.
"""

import re
from typing import List, Union
from urllib.parse import urlparse

class ContentExtractor:
	def __init__(self):
		"""
		Initialize the ContentExtractor.
		"""
		self.website_extractor = WebsiteExtractor()
		self.pdf_extractor = PDFExtractor()

	def is_url(self, source: str) -> bool:
		"""
		Check if the given source is a valid URL.

		Args:
			source (str): The source to check.

		Returns:
			bool: True if the source is a valid URL, False otherwise.
		"""
		try:
			# If the source doesn't start with a scheme, add 'https://'
			if not source.startswith(('http://', 'https://')):
				source = 'https://' + source

			result = urlparse(source)
			return all([result.scheme, result.netloc])
		except ValueError:
			return False

	def extract_content(self, source: str) -> str:
		"""
		Extract content from various sources.

		Args:
			source (str): URL or file path of the content source.

		Returns:
			str: Extracted text content.

		Raises:
			ValueError: If the source type is unsupported.
		"""
		try:
			if source.lower().endswith('.pdf'):
				return self.pdf_extractor.extract_content(source)
			elif self.is_url(source):
				return self.website_extractor.extract_content(source)
			else:
				raise ValueError("Unsupported source type")
		except Exception as e:
			logger.error(f"Error extracting content from {source}: {str(e)}")
			raise