import os
import logging
import requests
import re
import time
import hashlib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "ai-news-mcp-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# API Keys provided by user - Multiple sources for failover
NEWSDATA_API_KEY = "pub_ec96fa87f08243589ce78b1e236f57db"
NEWSAPI_KEY = "d4c92e160722416ba2e1782022053c86"
GNEWS_API_KEY = "14070ffb7f65ac62046967f94b155818"
THENEWSAPI_KEY = "kX4Wa4udZDVTd0GFik421KLfNVSmPkMVfteNdOhj"
CURRENTSAPI_KEY = "UH0NKr5z5ESol5Pn3UhFyNWITMnCjMGyflDo0yD-tFK4YsQp"
GUARDIAN_API_KEY = "3d6eea56-e0dc-46db-bbc8-62299f3449bd"

# MCP Server Configuration
OWNER_PHONE_NUMBER = "919701133665"
VALID_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "mcp-news-token-secure-2024")

class NewsService:
    def __init__(self):
        self.newsdata_url = "https://newsdata.io/api/1/news"
        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.gnews_url = "https://gnews.io/api/v4/search"
        self.thenewsapi_url = "https://api.thenewsapi.com/v1/news/all"
        self.currentsapi_url = "https://api.currentsapi.services/v1/latest-news"
        self.guardian_url = "https://content.guardianapis.com/search"
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.last_request_time = 0
        self.min_request_interval = 1  # 1 second between requests

    def fetch_news(self, keyword: Optional[str] = None, category: Optional[str] = None, country: Optional[str] = None, language: str = "en") -> List[Dict]:
        """Fetch news from multiple APIs with failover support and caching"""
        # Create cache key
        cache_key = hashlib.md5(f"{keyword}_{category}_{country}_{language}".encode()).hexdigest()

        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_duration):
                logger.info(f"Returning cached results for query: {keyword or category or 'default'}")
                return cached_data

        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - self.last_request_time))

        articles = []

        # Try NewsData.io first
        try:
            articles = self._fetch_from_newsdata(keyword, category, country, language)
            if articles:
                logger.info(f"Successfully fetched {len(articles)} articles from NewsData.io")
                self.cache[cache_key] = (articles, datetime.now())
                self.last_request_time = time.time()
                return articles
        except Exception as e:
            logger.warning(f"NewsData.io failed: {str(e)}")

        # Fallback to NewsAPI
        try:
            time.sleep(1)  # Brief delay between API attempts
            articles = self._fetch_from_newsapi(keyword, category, country, language)
            if articles:
                logger.info(f"Successfully fetched {len(articles)} articles from NewsAPI")
                self.cache[cache_key] = (articles, datetime.now())
                self.last_request_time = time.time()
                return articles
        except Exception as e:
            logger.warning(f"NewsAPI failed: {str(e)}")

        # Fallback to GNews
        try:
            time.sleep(1)  # Brief delay between API attempts
            articles = self._fetch_from_gnews(keyword, category, country, language)
            if articles:
                logger.info(f"Successfully fetched {len(articles)} articles from GNews")
                self.cache[cache_key] = (articles, datetime.now())
                self.last_request_time = time.time()
                return articles
        except Exception as e:
            logger.warning(f"GNews failed: {str(e)}")

        # Fallback to TheNewsAPI
        try:
            time.sleep(1)
            articles = self._fetch_from_thenewsapi(keyword, category, country, language)
            if articles:
                logger.info(f"Successfully fetched {len(articles)} articles from TheNewsAPI")
                self.cache[cache_key] = (articles, datetime.now())
                self.last_request_time = time.time()
                return articles
        except Exception as e:
            logger.warning(f"TheNewsAPI failed: {str(e)}")

        # Fallback to CurrentsAPI
        try:
            time.sleep(1)
            articles = self._fetch_from_currentsapi(keyword, category, country, language)
            if articles:
                logger.info(f"Successfully fetched {len(articles)} articles from CurrentsAPI")
                self.cache[cache_key] = (articles, datetime.now())
                self.last_request_time = time.time()
                return articles
        except Exception as e:
            logger.warning(f"CurrentsAPI failed: {str(e)}")

        # Final fallback to Guardian API
        try:
            time.sleep(1)
            articles = self._fetch_from_guardian(keyword, category, country, language)
            if articles:
                logger.info(f"Successfully fetched {len(articles)} articles from Guardian API")
                self.cache[cache_key] = (articles, datetime.now())
                self.last_request_time = time.time()
                return articles
        except Exception as e:
            logger.warning(f"Guardian API failed: {str(e)}")

        logger.error("All 6 news APIs failed")
        return []

    def _fetch_from_newsdata(self, keyword: Optional[str], category: Optional[str], country: Optional[str], language: str) -> List[Dict]:
        """Fetch news from NewsData.io API"""
        params = {
            "apikey": NEWSDATA_API_KEY,
            "language": language,
            "size": 10
        }

        if keyword:
            params["q"] = keyword
        if category:
            params["category"] = category
        if country:
            params["country"] = country

        response = requests.get(self.newsdata_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("status") == "success" and data.get("results"):
            articles = []
            for item in data["results"]:
                article = {
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", ""),
                    "source": item.get("source_id", ""),
                    "publishedAt": item.get("pubDate", ""),
                    "url": item.get("link", "")
                }
                articles.append(article)

            # Sort by published date
            return sorted(articles, key=lambda x: x.get("publishedAt", ""), reverse=True)

        return []

    def _fetch_from_newsapi(self, keyword: Optional[str], category: Optional[str], country: Optional[str], language: str) -> List[Dict]:
        """Fetch news from NewsAPI"""
        params = {
            "apiKey": NEWSAPI_KEY,
            "language": language,
            "pageSize": 10,
            "sortBy": "publishedAt"
        }

        if keyword:
            params["q"] = keyword

        response = requests.get(self.newsapi_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("status") == "ok" and data.get("articles"):
            articles = []
            for item in data["articles"]:
                article = {
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", ""),
                    "source": item.get("source", {}).get("name", ""),
                    "publishedAt": item.get("publishedAt", ""),
                    "url": item.get("url", "")
                }
                articles.append(article)

            return articles

        return []

    def _fetch_from_gnews(self, keyword: Optional[str], category: Optional[str], country: Optional[str], language: str) -> List[Dict]:
        """Fetch news from GNews API"""
        params = {
            "token": GNEWS_API_KEY,
            "lang": language,
            "max": 10,
            "sortby": "publishedAt"
        }

        if keyword:
            params["q"] = keyword
        if country:
            params["country"] = country

        response = requests.get(self.gnews_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("articles"):
            articles = []
            for item in data["articles"]:
                article = {
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", ""),
                    "source": item.get("source", {}).get("name", ""),
                    "publishedAt": item.get("publishedAt", ""),
                    "url": item.get("url", "")
                }
                articles.append(article)

            return articles

        return []

    def _fetch_from_thenewsapi(self, keyword: Optional[str], category: Optional[str], country: Optional[str], language: str) -> List[Dict]:
        """Fetch news from TheNewsAPI"""
        params = {
            "api_token": THENEWSAPI_KEY,
            "language": language,
            "limit": 10
        }

        if keyword:
            params["search"] = keyword
        if category:
            params["categories"] = category

        response = requests.get(self.thenewsapi_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("data"):
            articles = []
            for item in data["data"]:
                article = {
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("snippet", ""),
                    "source": item.get("source", ""),
                    "publishedAt": item.get("published_at", ""),
                    "url": item.get("url", "")
                }
                articles.append(article)
            return articles

        return []

    def _fetch_from_currentsapi(self, keyword: Optional[str], category: Optional[str], country: Optional[str], language: str) -> List[Dict]:
        """Fetch news from CurrentsAPI"""
        params = {
            "apiKey": CURRENTSAPI_KEY,
            "language": language[:2],  # CurrentsAPI uses 2-letter codes
            "limit": 10
        }

        if keyword:
            params["keywords"] = keyword
        if category:
            params["category"] = category
        if country:
            params["country"] = country[:2]

        response = requests.get(self.currentsapi_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("status") == "ok" and data.get("news"):
            articles = []
            for item in data["news"]:
                article = {
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", ""),
                    "source": item.get("author", ""),
                    "publishedAt": item.get("published", ""),
                    "url": item.get("url", "")
                }
                articles.append(article)
            return articles

        return []

    def _fetch_from_guardian(self, keyword: Optional[str], category: Optional[str], country: Optional[str], language: str) -> List[Dict]:
        """Fetch news from Guardian API"""
        params = {
            "api-key": GUARDIAN_API_KEY,
            "page-size": 10,
            "order-by": "newest",
            "show-fields": "headline,body,byline"
        }

        if keyword:
            params["q"] = keyword
        if category:
            params["section"] = category

        response = requests.get(self.guardian_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("response", {}).get("status") == "ok" and data.get("response", {}).get("results"):
            articles = []
            for item in data["response"]["results"]:
                fields = item.get("fields", {})
                article = {
                    "title": fields.get("headline", item.get("webTitle", "")),
                    "description": fields.get("body", "")[:200] + "..." if fields.get("body") else "",
                    "content": fields.get("body", ""),
                    "source": "The Guardian",
                    "publishedAt": item.get("webPublicationDate", ""),
                    "url": item.get("webUrl", "")
                }
                articles.append(article)
            return articles

        return []

class AIProcessor:
    """Simple AI processor for summarization and sentiment analysis without heavy dependencies"""

    def summarize_text(self, text: str) -> str:
        """Simple text summarization by extracting first 2 sentences"""
        if not text:
            return ""

        # Remove extra whitespace and clean text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())

        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)

        # Take first 2 meaningful sentences
        summary_sentences = []
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignore very short sentences
                summary_sentences.append(sentence)
                if len(summary_sentences) >= 2:
                    break

        if summary_sentences:
            summary = '. '.join(summary_sentences)
            if not summary.endswith('.'):
                summary += '.'
            return summary

        # Fallback: return first 150 characters
        return (cleaned_text[:150] + "...") if len(cleaned_text) > 150 else cleaned_text

    def analyze_sentiment(self, text: str) -> Dict[str, str]:
        """Simple sentiment analysis based on keywords"""
        if not text:
            return {"label": "neutral", "emoji": "😐"}

        text_lower = text.lower()

        # Positive keywords
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                         'success', 'win', 'victory', 'breakthrough', 'achievement', 'growth',
                         'profit', 'gain', 'rise', 'increase', 'improve', 'better', 'best',
                         'happy', 'joy', 'celebration', 'milestone', 'record', 'boost']

        # Negative keywords
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disaster', 'crisis',
                         'failure', 'loss', 'defeat', 'decline', 'crash', 'fall', 'drop',
                         'decrease', 'worse', 'worst', 'sad', 'tragedy', 'concern', 'worry',
                         'problem', 'issue', 'threat', 'risk', 'danger', 'warn', 'alert']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return {"label": "positive", "emoji": "😊"}
        elif negative_count > positive_count:
            return {"label": "negative", "emoji": "😠"}
        else:
            return {"label": "neutral", "emoji": "😐"}

class NLPProcessor:
    """Natural language processing for extracting parameters from queries"""

    def __init__(self):
        self.category_keywords = {
            'technology': ['tech', 'technology', 'ai', 'artificial intelligence', 'computer', 'software'],
            'sports': ['sports', 'football', 'basketball', 'cricket', 'soccer', 'game'],
            'health': ['health', 'medical', 'medicine', 'healthcare', 'disease', 'treatment'],
            'business': ['business', 'economy', 'finance', 'market', 'stock', 'company'],
            'entertainment': ['entertainment', 'movie', 'celebrity', 'music', 'film'],
            'science': ['science', 'research', 'study', 'discovery', 'scientific'],
            'politics': ['politics', 'government', 'election', 'policy', 'political']
        }

        self.country_keywords = {
            'us': ['usa', 'america', 'united states', 'american'],
            'in': ['india', 'indian'],
            'fr': ['france', 'french'],
            'de': ['germany', 'german'],
            'gb': ['britain', 'british', 'england', 'uk'],
            'ca': ['canada', 'canadian'],
            'au': ['australia', 'australian']
        }

    def extract_parameters(self, query_text: str) -> Dict[str, Optional[str]]:
        """Extract parameters from natural language query"""
        if not query_text:
            return {}

        query_lower = query_text.lower()
        extracted = {}

        # Extract category
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    extracted['category'] = category
                    break
            if 'category' in extracted:
                break

        # Extract country
        for country_code, keywords in self.country_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    extracted['country'] = country_code
                    break
            if 'country' in extracted:
                break

        # Extract keyword (simple approach)
        # Remove common words and extract meaningful terms
        common_words = ['news', 'latest', 'get', 'give', 'me', 'show', 'about', 'on', 'the', 'a', 'an']
        words = query_lower.split()
        keywords = [word for word in words if word not in common_words and len(word) > 2]

        if keywords and 'category' not in extracted:
            # Use first meaningful word as keyword if no category found
            extracted['keyword'] = keywords[0]

        return extracted

# Initialize services
news_service = NewsService()
ai_processor = AIProcessor()
nlp_processor = NLPProcessor()

def validate_params(keyword: Optional[str] = None, category: Optional[str] = None, country: Optional[str] = None, language: Optional[str] = None) -> Dict:
    """Validate query parameters"""
    errors = []

    valid_categories = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology', 'politics']
    valid_countries = ['us', 'gb', 'ca', 'au', 'in', 'fr', 'de', 'jp', 'br', 'mx', 'it', 'es', 'ru', 'cn']
    valid_languages = ['en', 'hi', 'te', 'fr', 'es', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'ar']

    if category and category.lower() not in valid_categories:
        errors.append(f"Invalid category '{category}'. Valid: {', '.join(valid_categories)}")

    if country and country.lower() not in valid_countries:
        errors.append(f"Invalid country '{country}'. Valid: {', '.join(valid_countries)}")

    if language and language.lower() not in valid_languages:
        errors.append(f"Invalid language '{language}'. Valid: {', '.join(valid_languages)}")

    return {'valid': len(errors) == 0, 'errors': errors}

def check_authorization():
    """Check if request has valid bearer token for MCP tools"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return False

    if auth_header.startswith('Bearer '):
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        return token == VALID_BEARER_TOKEN

    return False

# MCP Protocol Endpoints

@app.route('/mcp', methods=['POST'])
def mcp_handler():
    """Main MCP protocol handler"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Invalid JSON request"
            }), 400

        method = data.get('method')
        params = data.get('params', {})

        if method == 'tools/list':
            response = jsonify({
                "tools": [
                    {
                        "name": "validate",
                        "description": "Validate bearer token and return owner phone number",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "token": {
                                    "type": "string",
                                    "description": "Bearer token to validate"
                                }
                            },
                            "required": ["token"]
                        }
                    },
                    {
                        "name": "get_news",
                        "description": "Get AI-processed news articles with summarization and sentiment analysis",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "keyword": {
                                    "type": "string",
                                    "description": "Search keyword for news articles"
                                },
                                "category": {
                                    "type": "string",
                                    "description": "News category (business, entertainment, health, science, sports, technology, politics)"
                                },
                                "country": {
                                    "type": "string",
                                    "description": "Country code (us, gb, ca, au, in, fr, de, etc.)"
                                },
                                "language": {
                                    "type": "string",
                                    "description": "Language code (en, fr, es, de, etc.)",
                                    "default": "en"
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Natural language query for news search"
                                }
                            }
                        }
                    }
                ]
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response


        elif method == 'tools/call':
            tool_name = params.get('name')
            arguments = params.get('arguments', {})

            if tool_name == 'validate':
                token = arguments.get('token')
                logger.info(f"MCP Validation attempt - Token provided: {'Yes' if token else 'No'}")
                logger.info(f"Expected token: {VALID_BEARER_TOKEN}")
                logger.info(f"Received token: {token}")

                if token == VALID_BEARER_TOKEN:
                    logger.info(f"✅ MCP Validation SUCCESS - Returning phone: {OWNER_PHONE_NUMBER}")
                    response = jsonify({
                        "content": [
                            {
                                "type": "text",
                                "text": OWNER_PHONE_NUMBER
                            }
                        ]
                    })
                else:
                    logger.error(f"❌ MCP Validation FAILED - Invalid token: {token}")
                    response = jsonify({
                        "error": "Invalid token"
                    }), 401
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response

            elif tool_name == 'get_news':
                # Extract parameters
                keyword = arguments.get('keyword')
                category = arguments.get('category')
                country = arguments.get('country')
                language = arguments.get('language', 'en')
                query_text = arguments.get('query')

                # Handle natural language queries
                if query_text:
                    extracted_params = nlp_processor.extract_parameters(query_text)
                    keyword = extracted_params.get('keyword') or keyword
                    category = extracted_params.get('category') or category
                    country = extracted_params.get('country') or country

                # Set defaults
                if not keyword and not category:
                    keyword = "technology"

                # Validate parameters
                validation_result = validate_params(keyword, category, country, language)
                if not validation_result['valid']:
                    response = jsonify({
                        "error": "Invalid parameters",
                        "details": validation_result['errors']
                    }), 400
                    response.headers.add('Access-Control-Allow-Origin', '*')
                    return response


                # Fetch news articles
                logger.info(f"Fetching news with params: keyword={keyword}, category={category}, country={country}, language={language}")
                articles = news_service.fetch_news(
                    keyword=keyword,
                    category=category,
                    country=country,
                    language=language
                )

                if not articles:
                    response = jsonify({
                        "content": [
                            {
                                "type": "text",
                                "text": "No news articles found for the given parameters."
                            }
                        ]
                    })
                    response.headers.add('Access-Control-Allow-Origin', '*')
                    return response


                # Process articles with AI
                processed_articles = []
                for article in articles:
                    try:
                        # Generate summary
                        text_content = article.get('description', '') or article.get('content', '') or article.get('title', '')
                        summary = ai_processor.summarize_text(text_content)

                        # Analyze sentiment
                        sentiment_result = ai_processor.analyze_sentiment(summary)

                        processed_article = {
                            "title": article.get('title', ''),
                            "summary": summary,
                            "sentiment": {
                                "label": sentiment_result['label'],
                                "emoji": sentiment_result['emoji']
                            },
                            "source": article.get('source', ''),
                            "publishedAt": article.get('publishedAt', ''),
                            "language": language
                        }
                        processed_articles.append(processed_article)

                    except Exception as e:
                        logger.error(f"Error processing article: {str(e)}")
                        # Add article with minimal processing if AI fails
                        processed_articles.append({
                            "title": article.get('title', ''),
                            "summary": article.get('description', '')[:200] + "..." if article.get('description') else "",
                            "sentiment": {
                                "label": "neutral",
                                "emoji": "😐"
                            },
                            "source": article.get('source', ''),
                            "publishedAt": article.get('publishedAt', ''),
                            "language": language
                        })

                # Format response for MCP
                response_text = "📰 **Latest News Articles**\n\n"
                for i, article in enumerate(processed_articles[:5], 1):
                    response_text += f"**{i}. {article['title']}**\n"
                    response_text += f"🏢 *{article['source']}* | {article['sentiment']['emoji']} *{article['sentiment']['label']}*\n"
                    response_text += f"📝 {article['summary']}\n"
                    response_text += f"🕐 {article['publishedAt']}\n\n"

                response = jsonify({
                    "content": [
                        {
                            "type": "text",
                            "text": response_text
                        }
                    ]
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response


            else:
                response = jsonify({
                    "error": f"Unknown tool: {tool_name}"
                }), 400
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response

        else:
            response = jsonify({
                "error": f"Unknown method: {method}"
            }), 400
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response


    except Exception as e:
        logger.error(f"Error in MCP handler: {str(e)}")
        response = jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


# Legacy REST API Endpoints (for backward compatibility)

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    response = jsonify({
        "service": "AI News MCP Server",
        "version": "1.0.0",
        "description": "Real-time news aggregation with AI summarization and sentiment analysis",
        "mcp_endpoint": "/mcp",
        "owner_phone": OWNER_PHONE_NUMBER,
        "endpoints": {
            "/health": "Server health check",
            "/news": "Get news with AI processing (supports keyword, category, country, language, q params)",
            "/mcp": "MCP protocol endpoint for Puch AI integration"
        },
        "examples": {
            "technology_news": "/news?keyword=AI",
            "sports_news": "/news?category=sports",
            "natural_language": "/news?q=latest technology news"
        }
    }), 200
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        response = jsonify({
            "status": "healthy",
            "service": "AI News MCP Server",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mcp_ready": True,
            "owner_phone": OWNER_PHONE_NUMBER
        }), 200
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        response = jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/news', methods=['GET'])
def get_news():
    """Main news endpoint supporting both query params and natural language"""
    try:
        # Get query parameters
        keyword = request.args.get('keyword')
        category = request.args.get('category')
        country = request.args.get('country')
        language = request.args.get('language', 'en')

        # Handle natural language queries
        query_text = request.args.get('q') or request.args.get('query')

        if query_text:
            # Process natural language query
            extracted_params = nlp_processor.extract_parameters(query_text)
            keyword = extracted_params.get('keyword') or keyword
            category = extracted_params.get('category') or category
            country = extracted_params.get('country') or country

        # Set defaults
        if not keyword and not category:
            keyword = "technology"

        # Validate parameters
        validation_result = validate_params(keyword, category, country, language)
        if not validation_result['valid']:
            response = jsonify({
                "error": "Invalid parameters",
                "details": validation_result['errors']
            }), 400
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response


        # Fetch news articles
        logger.info(f"Fetching news with params: keyword={keyword}, category={category}, country={country}, language={language}")
        articles = news_service.fetch_news(
            keyword=keyword,
            category=category,
            country=country,
            language=language
        )

        if not articles:
            response = jsonify([]), 200
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response


        # Process articles with AI
        processed_articles = []
        for article in articles:
            try:
                # Generate summary
                text_content = article.get('description', '') or article.get('content', '') or article.get('title', '')
                summary = ai_processor.summarize_text(text_content)

                # Analyze sentiment
                sentiment_result = ai_processor.analyze_sentiment(summary)

                processed_article = {
                    "title": article.get('title', ''),
                    "summary": summary,
                    "sentiment": {
                        "label": sentiment_result['label'],
                        "emoji": sentiment_result['emoji']
                    },
                    "source": article.get('source', ''),
                    "publishedAt": article.get('publishedAt', ''),
                    "language": language
                }
                processed_articles.append(processed_article)

            except Exception as e:
                logger.error(f"Error processing article: {str(e)}")
                # Add article with minimal processing if AI fails
                processed_articles.append({
                    "title": article.get('title', ''),
                    "summary": article.get('description', '')[:200] + "..." if article.get('description') else "",
                    "sentiment": {
                        "label": "neutral",
                        "emoji": "😐"
                    },
                    "source": article.get('source', ''),
                    "publishedAt": article.get('publishedAt', ''),
                    "language": language
                })

        response = jsonify(processed_articles), 200
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


    except Exception as e:
        logger.error(f"Error in get_news endpoint: {str(e)}")
        response = jsonify({
            "error": "Internal server error",
            "message": "Failed to fetch and process news articles",
            "details": str(e)
        }), 500
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/getnews', methods=['GET'])
def get_news_command():
    """Command-style endpoint for backwards compatibility"""
    return get_news()

@app.errorhandler(404)
def not_found(error):
    response = jsonify({
        "error": "Endpoint not found",
        "message": "Available endpoints: /news, /getnews, /health, /mcp"
    }), 404
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.errorhandler(500)
def internal_error(error):
    response = jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)