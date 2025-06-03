"""
Vibe classification using NLP analysis of captions and metadata
"""
import re
import json
from typing import List, Dict, Set
import logging
from collections import Counter
import spacy
from transformers import pipeline

from ..config import VIBE_KEYWORDS, NLP_MODEL

logger = logging.getLogger(__name__)


class VibeClassifier:
    """Classifies fashion vibes based on text content and metadata"""
    
    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self.load_models()
    
    def load_models(self):
        """Load NLP models"""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic text processing")
                self.nlp = None
            
            # Load sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            logger.info("Loaded NLP models")
            
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags symbols but keep the text
        text = re.sub(r'[@#]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        text = self.preprocess_text(text)
        
        if self.nlp:
            # Use spaCy for better keyword extraction
            doc = self.nlp(text)
            keywords = []
            
            for token in doc:
                # Extract nouns, adjectives, and relevant words
                if (token.pos_ in ['NOUN', 'ADJ'] and 
                    len(token.text) > 2 and 
                    not token.is_stop and 
                    not token.is_punct):
                    keywords.append(token.lemma_)
            
            return keywords
        else:
            # Basic keyword extraction
            words = text.split()
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in words if len(word) > 2 and word not in stop_words]
            return keywords
    
    def calculate_vibe_scores(self, text: str) -> Dict[str, float]:
        """Calculate scores for each vibe based on keyword matching"""
        keywords = self.extract_keywords(text)
        vibe_scores = {}
        
        for vibe, vibe_keywords in VIBE_KEYWORDS.items():
            score = 0
            matches = []
            
            for keyword in keywords:
                for vibe_keyword in vibe_keywords:
                    # Check for exact match or partial match
                    if (keyword == vibe_keyword or 
                        vibe_keyword in keyword or 
                        keyword in vibe_keyword):
                        score += 1
                        matches.append(keyword)
            
            # Normalize score by text length
            if keywords:
                normalized_score = score / len(keywords)
            else:
                normalized_score = 0
            
            vibe_scores[vibe] = {
                'score': normalized_score,
                'raw_score': score,
                'matches': matches
            }
        
        return vibe_scores
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if not self.sentiment_analyzer or not text:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        try:
            results = self.sentiment_analyzer(text[:512])  # Limit text length
            
            # Get the highest scoring sentiment
            best_sentiment = max(results[0], key=lambda x: x['score'])
            
            return {
                "sentiment": best_sentiment['label'].lower(),
                "confidence": best_sentiment['score']
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def extract_fashion_context(self, metadata: Dict) -> Dict:
        """Extract fashion-related context from Instagram metadata"""
        context = {
            "hashtags": [],
            "mentions": [],
            "location": None,
            "engagement": {}
        }
        
        try:
            if "instagram_data" in metadata:
                ig_data = metadata["instagram_data"]
                
                # Extract from caption
                if "node" in ig_data:
                    node = ig_data["node"]
                    
                    # Get caption text
                    if "edge_media_to_caption" in node:
                        edges = node["edge_media_to_caption"]["edges"]
                        if edges:
                            caption_text = edges[0]["node"]["text"]
                            
                            # Extract hashtags
                            hashtags = re.findall(r'#(\w+)', caption_text)
                            context["hashtags"] = hashtags
                            
                            # Extract mentions
                            mentions = re.findall(r'@(\w+)', caption_text)
                            context["mentions"] = mentions
                    
                    # Get engagement metrics
                    if "edge_liked_by" in node:
                        context["engagement"]["likes"] = node["edge_liked_by"]["count"]
                    
                    if "edge_media_to_comment" in node:
                        context["engagement"]["comments"] = node["edge_media_to_comment"]["count"]
                    
                    # Get location
                    if "location" in node and node["location"]:
                        context["location"] = node["location"]
        
        except Exception as e:
            logger.error(f"Error extracting fashion context: {e}")
        
        return context
    
    def classify_vibes(self, metadata: Dict, max_vibes: int = 3) -> List[str]:
        """Classify vibes for a video based on metadata and captions"""
        try:
            # Combine all text sources
            text_sources = []
            
            # Add caption from text file
            if "caption" in metadata:
                text_sources.append(metadata["caption"])
            
            # Add Instagram caption
            if "instagram_data" in metadata:
                ig_data = metadata["instagram_data"]
                if "node" in ig_data and "edge_media_to_caption" in ig_data["node"]:
                    edges = ig_data["node"]["edge_media_to_caption"]["edges"]
                    if edges:
                        text_sources.append(edges[0]["node"]["text"])
            
            # Combine all text
            combined_text = " ".join(text_sources)
            
            if not combined_text.strip():
                logger.warning("No text content found for vibe classification")
                return []
            
            # Calculate vibe scores
            vibe_scores = self.calculate_vibe_scores(combined_text)
            
            # Extract fashion context for additional signals
            context = self.extract_fashion_context(metadata)
            
            # Boost scores based on hashtags
            for hashtag in context["hashtags"]:
                hashtag_lower = hashtag.lower()
                for vibe, vibe_keywords in VIBE_KEYWORDS.items():
                    if hashtag_lower in [kw.lower() for kw in vibe_keywords]:
                        if vibe in vibe_scores:
                            vibe_scores[vibe]['score'] += 0.2  # Boost for hashtag match
            
            # Sort vibes by score
            sorted_vibes = sorted(
                vibe_scores.items(), 
                key=lambda x: x[1]['score'], 
                reverse=True
            )
            
            # Select top vibes with minimum threshold
            selected_vibes = []
            min_threshold = 0.1
            
            for vibe, score_data in sorted_vibes[:max_vibes]:
                if score_data['score'] >= min_threshold:
                    selected_vibes.append(vibe)
            
            # If no vibes meet threshold, select the top one if it has any matches
            if not selected_vibes and sorted_vibes:
                top_vibe, top_score = sorted_vibes[0]
                if top_score['raw_score'] > 0:
                    selected_vibes.append(top_vibe)
            
            logger.info(f"Classified vibes: {selected_vibes}")
            return selected_vibes
            
        except Exception as e:
            logger.error(f"Error in vibe classification: {e}")
            return []
    
    def get_vibe_explanation(self, vibe: str, metadata: Dict) -> Dict:
        """Get explanation for why a vibe was classified"""
        text_sources = []
        
        if "caption" in metadata:
            text_sources.append(metadata["caption"])
        
        if "instagram_data" in metadata:
            ig_data = metadata["instagram_data"]
            if "node" in ig_data and "edge_media_to_caption" in ig_data["node"]:
                edges = ig_data["node"]["edge_media_to_caption"]["edges"]
                if edges:
                    text_sources.append(edges[0]["node"]["text"])
        
        combined_text = " ".join(text_sources)
        vibe_scores = self.calculate_vibe_scores(combined_text)
        
        if vibe in vibe_scores:
            return {
                "vibe": vibe,
                "score": vibe_scores[vibe]['score'],
                "matches": vibe_scores[vibe]['matches'],
                "keywords": VIBE_KEYWORDS[vibe]
            }
        
        return {}
