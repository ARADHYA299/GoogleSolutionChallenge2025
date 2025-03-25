import plotly.graph_objects as go
import matplotlib.pyplot as plt 
from datetime import datetime
import os
import re
os.system("pip install --upgrade transformers accelerate torch")
import torch
import gradio as gr 
import pandas as pd
import numpy as np
import yfinance as yf
device = 0 if torch.cuda.is_available() else -1
print("pip install nltk")
import nltk
print("pip install transformers")

from transformers import pipeline

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# Get HF API token from environment
import requests
# Get HF API token from environment
HF_API_KEY = os.environ.get("tutorhfkey")
if not HF_API_KEY:
    print("WARNING: No Hugging Face API token found. API calls will likely fail.")
    print("Set the 'tutorhfkey' environment variable in your Hugging Face Space settings.")
SYSTEM_INSTRUCTIONS = """You are a financial markets tutor , assistant and adviser designed to educate and recommend/advise beginners, intermediates or experts about investing, financial instruments, money markets and market dynamics.
Answer briefly and point by point and to the point, don't explain unnecessarily, and use things like real life examples or the things with which they can relate them.
Things to remember:
-to first ask the user how much do they know about the financial and money market(if they dont want this)
-whether they want to know about a specific security (stocks,commodities and others)
Capabilities:
- Teach financial concepts in an interactive and engaging way.
- Guide users on different types of financial markets (stocks, bonds, crypto, commodities, etc.).
- Explain investment strategies, risk management, and portfolio diversification but tell these only when asked.
- Answer questions related to fundamental and technical analysis.
- If they give you certain budgets provide them the best areas to invest along with risk involved but warn them about the risks and also advise to think before investing.
- You can analyze user sentiment to provide personalized financial advice.
- You can explain stock trends and market news sentiment when requested.
- If the user asks any other questions than finance , politely tell them that you are not able to answer or you are not designed for that purpose

Guidelines:
- Begin by asking whats your level of knowledge like beginner , intermediate and expert.
- continue by understanding the user's financial knowledge level or by asking them what level of knowledge they have about financial and money markets.
- let the user control the flow and ask after every concept whether they learned or not.
- If the user is new/beginner, first ask them which financial market they are interested in.
- Provide structured/point by point explanations with real life or related examples.
- After a major concept ask them if they understood or do they have any kind of doubt regarding the topic.
- Be as user friendly as possible.
- Use simple language and try to give response in an example or things with which the user can relate.
- If the user seems anxious or negative about investing, provide reassuring advice about risk management.
- If the user seems overconfident, remind them about market volatility and risk assessment."""

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def generate_text_with_api(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """Generate text using the Hugging Face API with Mistral-7B-Instruct model."""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    # Ensure prompt is correctly formatted for Mistral
    if not prompt.startswith("<s>[INST]"):
        prompt = f"<s>[INST]{prompt}[/INST]"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False  # Changed to False to get only the new text
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        # Check if we got a valid response
        if response.status_code != 200:
            print(f"API Error: Status code {response.status_code}")
            print(f"Response content: {response.text}")
            return f"Sorry, I couldn't generate a response. API error: {response.status_code}"
            
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                print(f"Unexpected API response format: {result}")
                return "I'm sorry, I couldn't generate a helpful response right now."
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        else:
            print(f"Unexpected API response format: {result}")
            return "I'm sorry, I couldn't generate a helpful response right now."
    except Exception as e:
        print(f"API Error: {str(e)}")
        return f"I'm sorry, I couldn't generate a response due to an API error. Please try again later."

        
# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sentiments = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",device = device)
finance_sentiment = SentimentIntensityAnalyzer()




NEWS_API_KEY = os.environ.get("newsapikey", "38666Z58XHGQB9X6")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# System instructions for the financial tutor

def generate_response(prompt):
    """Generate text using the Mistral-7B-Instruct model via HF API."""
    try:
        # Add a fallback in case the API fails
        if not HF_API_KEY or HF_API_KEY == "":
            return "I'm sorry, but I need an API key to provide responses. Please set the 'tutorhfkey' environment variable."
            
        response = generate_text_with_api(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9)
        
        # Clean up response text
        response = response.strip()
            
        # If the response is empty or too short, provide a fallback
        if len(response) < 10:
            return "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."
            
        return response
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return "I'm sorry, I couldn't generate a response. Please try again."
        
def format_prompt(messages, sentiment_data=None, stock_data=None, news_sentiment=None):
    """Format the conversation history for the model with additional context"""
    prompt = SYSTEM_INSTRUCTIONS + "\n\n"

    # Add sentiment analysis context if available
    if sentiment_data:
        prompt += f"User Sentiment Analysis: The user's message appears to be {sentiment_data['sentiment'].lower()} "
        prompt += f"with confidence {sentiment_data['confidence']:.2f}. "
        prompt += f"Compound score: {sentiment_data['compound_score']:.2f}\n\n"

    # Add stock data context if available
    if isinstance(stock_data, dict) and 'ticker' in stock_data:
        prompt += f"Stock Analysis for {stock_data['ticker']}: "
        prompt += f"Current price: ${stock_data['current_price']:.2f}, "
        prompt += f"Weekly change: {stock_data['weekly_change']:.2f}%, "
        prompt += f"Monthly change: {stock_data['monthly_change']:.2f}%, "
        prompt += f"Volatility: {stock_data['volatility']:.2f}%\n\n"

    # Add news sentiment context if available
    if isinstance(news_sentiment, dict) and 'overall_sentiment' in news_sentiment:
        prompt += f"News Sentiment: Overall {news_sentiment['overall_sentiment'].lower()} "
        prompt += f"with score {news_sentiment['overall_score']:.2f}\n\n"

    # Add conversation history
    prompt += "Conversation history:\n"
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            prompt += f"User: {content}\n"
        else:
            prompt += f"Assistant: {content}\n"

    # Add current query instruction
    if messages and messages[-1]["role"] == "user":
        prompt += "\nPlease respond to the user's latest question in a helpful, concise manner focused on financial education."

    return prompt


def get_investment_recommendations(risk_tolerance, budget=None):
    """Generate investment recommendations based on risk tolerance and budget"""

    recommendations = {
        'very low': {
            'description': "For very low risk tolerance, focus on capital preservation with minimal volatility.",
            'allocation': [
                {'type': 'High-yield Savings', 'percentage': 40},
                {'type': 'Government Bonds', 'percentage': 35},
                {'type': 'Investment Grade Corporate Bonds', 'percentage': 20},
                {'type': 'Blue-chip Stocks', 'percentage': 5}
            ]
        },
        'low': {
            'description': "For low risk tolerance, prioritize safety with some income potential.",
            'allocation': [
                {'type': 'Government Bonds', 'percentage': 30},
                {'type': 'Corporate Bonds', 'percentage': 30},
                {'type': 'Blue-chip Stocks', 'percentage': 25},
                {'type': 'High-yield Savings', 'percentage': 10},
                {'type': 'REITs', 'percentage': 5}
            ]
        },
        'moderate': {
            'description': "For moderate risk tolerance, balance growth and income with diversification.",
            'allocation': [
                {'type': 'Index Funds', 'percentage': 40},
                {'type': 'Corporate Bonds', 'percentage': 25},
                {'type': 'Blue-chip Stocks', 'percentage': 20},
                {'type': 'International Stocks', 'percentage': 10},
                {'type': 'REITs', 'percentage': 5}
            ]
        },
        'high': {
            'description': "For high risk tolerance, focus on long-term growth with higher volatility.",
            'allocation': [
                {'type': 'Growth Stocks', 'percentage': 45},
                {'type': 'Index Funds', 'percentage': 25},
                {'type': 'International Stocks', 'percentage': 15},
                {'type': 'Corporate Bonds', 'percentage': 10},
                {'type': 'Alternative Investments', 'percentage': 5}
            ]
        },
        'very high': {
            'description': "For very high risk tolerance, maximize growth potential with significant volatility.",
            'allocation': [
                {'type': 'Growth Stocks', 'percentage': 40},
                {'type': 'Emerging Markets', 'percentage': 25},
                {'type': 'Small Cap Stocks', 'percentage': 20},
                {'type': 'Commodities', 'percentage': 10},
                {'type': 'Cryptocurrency', 'percentage': 5}
            ]
        }
    }

    result = recommendations.get(risk_tolerance.lower(), recommendations['moderate'])

    if budget and 'amount' in budget:
        amount = budget['amount']
        for item in result['allocation']:
            item['amount'] = amount * (item['percentage'] / 100)

    return result

# Extract stock ticker symbols from user message
def extract_tickers(message):
    # Look for standard stock ticker patterns (1-5 uppercase letters)
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    potential_tickers = re.findall(ticker_pattern, message)

    # Filter out common English words and acronyms that aren't tickers
    common_words = {'I', 'A', 'AN', 'THE', 'AND', 'OR', 'IF', 'IS', 'IT', 'BE', 'TO', 'IN', 'ON', 'AT', 'OF', 'FOR'}
    filtered_tickers = [ticker for ticker in potential_tickers if ticker not in common_words]

    return filtered_tickers

# Get news sentiment for a specific topic or ticker - Using a mock implementation since we don't have API key
def get_news_sentiment(query, max_results=5):
    try:
        # Mock implementation that returns simulated news sentiment
        sentiment_options = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        scores = np.random.normal(0, 0.5, 3)  # Generate random scores with normal distribution

        # Adjust scores to be between -1 and 1
        scores = np.clip(scores, -1, 1)

        # Create mock articles
        articles = []
        for i in range(min(3, max_results)):
            sentiment_score = scores[i]
            if sentiment_score >= 0.05:
                sentiment = "POSITIVE"
            elif sentiment_score <= -0.05:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"

            articles.append({
                'headline': f"Latest news about {query} - Article {i+1}",
                'sentiment': sentiment,
                'score': sentiment_score,
                'source': f"Financial Source {i+1}",
                'url': f"https://example.com/news/{i+1}"
            })

        # Calculate overall sentiment
        avg_score = sum(article['score'] for article in articles) / len(articles)

        if avg_score >= 0.05:
            overall = "POSITIVE"
        elif avg_score <= -0.05:
            overall = "NEGATIVE"
        else:
            overall = "NEUTRAL"

        return {
            'overall_sentiment': overall,
            'overall_score': avg_score,
            'articles': articles
        }
    except Exception as e:
        return f"Error analyzing news sentiment: {str(e)}"

def get_stock_data(ticker, period='1mo'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return f"No data found for ticker {ticker}. It may be delisted or incorrectly spelled."

        # Calculate some basic technical indicators
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()

        # Determine trend
        current_price = hist['Close'].iloc[-1]
        week_ago_price = hist['Close'].iloc[-6] if len(hist) >= 6 else hist['Close'].iloc[0]
        month_ago_price = hist['Close'].iloc[0]

        weekly_change = ((current_price - week_ago_price) / week_ago_price) * 100
        monthly_change = ((current_price - month_ago_price) / month_ago_price) * 100

        # Determine if stock is above or below moving averages
        above_sma20 = current_price > hist['SMA20'].iloc[-1] if not np.isnan(hist['SMA20'].iloc[-1]) else "Unknown"
        above_sma50 = current_price > hist['SMA50'].iloc[-1] if not np.isnan(hist['SMA50'].iloc[-1]) else "Unknown"
        hist['Daily_Return'] = hist['Close'].pct_change()
        volatility = hist['Daily_Return'].std() * 100  # Convert to percentage

        # Prepare chart data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Price'))
        if not hist['SMA20'].isnull().all():
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], mode='lines', name='SMA20'))
        if not hist['SMA50'].isnull().all():
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], mode='lines', name='SMA50'))

        fig.update_layout(
            title=f"{ticker} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            template="plotly_white"
        )

        return {
            'ticker': ticker,
            'current_price': current_price,
            'weekly_change': weekly_change,
            'monthly_change': monthly_change,
            'above_sma20': above_sma20,
            'above_sma50': above_sma50,
            'volatility': volatility,
            'chart': fig
        }

    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return f"Error analyzing {ticker}: {str(e)}"

 

# Analyze user message sentiment
def analyze_user_sentiment(message):
    try:
        # Use sentiment analysis pipeline
        result = sentiments(message)
        if not result:
            return None

        sentiment_result = result[0]
        sentiment_label = sentiment_result['label']
        confidence = sentiment_result['score']

        # Use NLTK VADER for more nuanced analysis
        vader_scores = finance_sentiment.polarity_scores(message)

        return {
            'sentiment': sentiment_label,
            'confidence': confidence,
            'compound_score': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu']
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return None

# Extract investment budget from message
def extract_budget(message):
    # Pattern for currency amounts
    budget_pattern = r'(\$|£|€|₹)?\s?(\d+[,.]?\d*)\s?(thousand|million|billion|k|m|b)?\s?(dollars|usd|rupees|inr|euros|eur|pounds|gbp)?'

    match = re.search(budget_pattern, message.lower())
    if match:
        amount = match.group(2).replace(',', '')
        multiplier = match.group(3) if match.group(3) else ''

        try:
            amount = float(amount)

            if multiplier:
                if multiplier.lower() in ['k', 'thousand']:
                    amount *= 1000
                elif multiplier.lower() in ['m', 'million']:
                    amount *= 1000000
                elif multiplier.lower() in ['b', 'billion']:
                    amount *= 1000000000

            currency = match.group(1) if match.group(1) else (match.group(4) if match.group(4) else 'USD')

            return {
                'amount': amount,
                'currency': currency
            }
        except:
            pass

    return None

# Extract risk tolerance from user message
def extract_risk_tolerance(message):
    msg = message.lower()

    risk_phrases = {
        'very low': ['very conservative', 'extremely safe', 'no risk', 'safest possible', 'can\'t afford to lose'],
        'low': ['conservative', 'safe', 'low risk', 'minimal risk', 'safety', 'secure'],
        'moderate': ['balanced', 'moderate', 'middle ground', 'medium risk', 'some risk'],
        'high': ['aggressive', 'high risk', 'risky', 'growth focused', 'willing to take risks'],
        'very high': ['very aggressive', 'highest risk', 'extremely risky', 'maximum growth', 'speculative']
    }

    for tolerance, phrases in risk_phrases.items():
        if any(phrase in msg for phrase in phrases):
            return tolerance

    return None

# Identify financial education level
def identify_education_level(message):
    msg = message.lower()

    if any(term in msg for term in ['beginner', 'new', 'novice', 'starting', 'basics', 'fundamental', 'never invested']):
        return 'beginner'
    elif any(term in msg for term in ['intermediate', 'some experience', 'familiar', 'understand']):
        return 'intermediate'
    elif any(term in msg for term in ['advanced', 'expert', 'professional', 'experienced', 'seasoned']):
        return 'advanced'

    return None

def respond(message, chat_history, conversation_state, user_profile):
    """Process user message and get response from the model"""

    # Initialize conversation if empty
    if not conversation_state:
        conversation_state = [
            {"role": "assistant", "content": "Hello! I'm your financial markets tutor. Before we begin, could you tell me your current level of knowledge about financial markets? Are you a beginner, intermediate, or advanced?"}
        ]

    # Add user message to conversation state
    conversation_state.append({"role": "user", "content": message})

    try:
        # Analyze user message
        sentiment_data = analyze_user_sentiment(message)
        education_level = identify_education_level(message)
        risk_tolerance = extract_risk_tolerance(message)
        budget = extract_budget(message)
        tickers = extract_tickers(message)

        # Update user profile
        if education_level:
            user_profile["education_level"] = education_level
        if risk_tolerance:
            user_profile["risk_tolerance"] = risk_tolerance
        if budget:
            user_profile["budget"] = budget

        # Process ticker information if found
        stock_data = None
        news_sentiment = None
        chart = None

        if tickers and (re.search(r'stock|price|ticker|trend|chart|analysis', message.lower()) or
                       any(ticker.lower() in message.lower() for ticker in tickers)):
            # Only process the first ticker for simplicity
            ticker = tickers[0]
            stock_data = get_stock_data(ticker)
            if isinstance(stock_data, dict) and 'chart' in stock_data:
                chart = stock_data['chart']
            news_sentiment = get_news_sentiment(ticker)

        # Get investment recommendations if user profile has enough information
        recommendations = None
        if "risk_tolerance" in user_profile and re.search(r'invest|allocation|portfolio|recommend|suggest|advice', message.lower()):
            recommendations = get_investment_recommendations(user_profile["risk_tolerance"], user_profile.get("budget", None))

        # Format the prompt with additional context
        prompt = format_prompt(conversation_state, sentiment_data, stock_data, news_sentiment)

        # Get response from model
        response_text = generate_response(prompt)
        
        # If response is empty or very short, provide fallback
        if len(response_text) < 10:
            response_text = "I'm sorry, I'm having trouble generating a response. Please try asking your question in a different way."

        # Enhance response with additional data if available
        if chart:
            response_text += "\n\n[Stock chart for " + stock_data['ticker'] + " is displayed separately]"

        # Add recommendations if available
        if recommendations:
            recommendation_text = f"\n\nBased on your {user_profile['risk_tolerance']} risk tolerance"
            if 'budget' in user_profile and 'amount' in user_profile['budget']:
                amount_str = f"{user_profile['budget']['amount']:,.2f}"
                recommendation_text += f" and budget of {amount_str}"
            recommendation_text += f", here's a suggested portfolio allocation:\n\n"
            recommendation_text += recommendations['description'] + "\n\n"

            # Format allocation as a table
            allocation_text = "Investment Type | Percentage"
            allocation_text += "\n--------------|----------"
            for item in recommendations['allocation']:
                allocation_text += f"\n{item['type']} | {item['percentage']}%"

            response_text += "\n\n" + recommendation_text + allocation_text

        # Add assistant response to conversation
        conversation_state.append({"role": "assistant", "content": response_text})

        # Update the chat history for Gradio display
        chat_history.append((message, response_text))
        
        return chat_history, conversation_state, user_profile, chart
    
    except Exception as e:
        print(f"Error in respond function: {str(e)}")
        error_message = "I'm sorry, I encountered an error while processing your request. Please try again or ask a different question."
        conversation_state.append({"role": "assistant", "content": error_message})
        chat_history.append((message, error_message))
        return chat_history, conversation_state, user_profile, None

def fallback_response(message):
    """Provide a basic response when the model fails"""
    if re.search(r'hello|hi|hey|greetings', message.lower()):
        return "Hello! I'm your financial markets tutor. How can I help you today?"
    
    if re.search(r'stock|price|ticker|trend|chart|analysis', message.lower()):
        return "I'd be happy to help you analyze stocks. Could you specify which ticker you're interested in?"
    
    if re.search(r'invest|allocation|portfolio|recommend|suggest|advice', message.lower()):
        return "For investment advice, I need to know your risk tolerance (conservative, moderate, or aggressive) and optionally your budget. Could you provide that information?"
    
    if re.search(r'beginner|new|start|learn|basics', message.lower()):
        return "As a beginner in financial markets, I recommend starting with understanding the basics of stocks, bonds, and mutual funds. Would you like me to explain any of these concepts?"
    
    return "I'm here to help with your financial questions. Could you provide more details about what you'd like to know about investing or financial markets?"
   

# Function to create personalized financial reports
def create_financial_report(user_profile):
    """Generate a personalized financial report based on user profile"""
    if not user_profile or "education_level" not in user_profile:
        return "Not enough user data collected to generate a personalized report."

    # Create a report header
    now = datetime.now()
    report = f"# Personal Financial Report\n\n"
    report += f"Generated on: {now.strftime('%B %d, %Y')}\n\n"

    # Add user profile information
    report += "## Your Profile\n\n"
    report += f"- Knowledge Level: {user_profile.get('education_level', 'Not specified')}\n"

    if "risk_tolerance" in user_profile:
        report += f"- Risk Tolerance: {user_profile.get('risk_tolerance', 'Not specified')}\n"

    if "budget" in user_profile and "amount" in user_profile["budget"]:
        amount = user_profile["budget"]["amount"]
        currency = user_profile["budget"].get("currency", "USD")
        report += f"- Investment Budget: {amount:,.2f} {currency}\n"

    # Add recommendations section if we have risk tolerance
    if "risk_tolerance" in user_profile:
        recommendations = get_investment_recommendations(user_profile["risk_tolerance"],
                                                         user_profile.get("budget", None))

        if recommendations:
            report += "\n## Investment Recommendations\n\n"
            report += recommendations["description"] + "\n\n"

            # Create a table of allocations
            report += "| Investment Type | Percentage |"
            if "budget" in user_profile and "amount" in user_profile["budget"]:
                report += " Amount |"
            report += "\n|-----------------|------------|"
            if "budget" in user_profile and "amount" in user_profile["budget"]:
                report += "--------|"
            report += "\n"

            for item in recommendations["allocation"]:
                report += f"| {item['type']} | {item['percentage']}% |"
                if "budget" in user_profile and "amount" in user_profile["budget"]:
                    amount = item.get("amount", 0)
                    report += f" {amount:,.2f} |"
                report += "\n"

    # Add educational resources based on level
    report += "\n## Recommended Educational Resources\n\n"

    if user_profile.get("education_level") == "beginner":
        report += "### For Beginners:\n\n"
        report += "1. **Investopedia Basics** - Fundamental concepts and terminology\n"
        report += "2. **Khan Academy: Personal Finance** - Free courses on investing basics\n"
        report += "3. **'A Random Walk Down Wall Street'** by Burton Malkiel - Classic book for beginners\n"
        report += "4. **Robinhood Learn** - Simple explanations of investment concepts\n"
    elif user_profile.get("education_level") == "intermediate":
        report += "### For Intermediate Investors:\n\n"
        report += "1. **'The Intelligent Investor'** by Benjamin Graham - Value investing principles\n"
        report += "2. **Morningstar Investment Classroom** - More advanced investment concepts\n"
        report += "3. **Yahoo Finance** - Research tools and market analysis\n"
        report += "4. **'The Little Book of Common Sense Investing'** by John Bogle\n"
    else:  # advanced
        report += "### For Advanced Investors:\n\n"
        report += "1. **Bloomberg Terminal** (if accessible) - Professional-grade research\n"
        report += "2. **CFA Institute Resources** - Professional investment analysis\n"
        report += "3. **'Security Analysis'** by Benjamin Graham and David Dodd\n"
        report += "4. **Journal of Finance** - Academic research on financial markets\n"

    # Add market outlook section
    report += "\n## Current Market Outlook\n\n"
    report += "This section would typically contain current market analysis and trends.\n"
    report += "For real-time and accurate market outlook, consider consulting financial news sources like:\n\n"
    report += "- Wall Street Journal\n"
    report += "- Financial Times\n"
    report += "- Bloomberg\n"
    report += "- CNBC\n"

    # Add disclaimer
    report += "\n## Disclaimer\n\n"
    report += "_This report is generated based on your interaction with the Financial Markets Tutor. "
    report += "It is for educational purposes only and does not constitute financial advice. "
    report += "Always consult with a qualified financial advisor before making investment decisions._\n"

    return report



def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 StockSensei 📊")
        gr.Markdown("The genie of financial markets is here to help you, Ask me anything about any market.")

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    value=[], 
                    show_label=False, 
                    height=500, 
                    bubble_full_width=False, 
                    show_copy_button=True,
                )
                conversation_state = gr.State([])
                user_profile = gr.State({})

                with gr.Row():
                    message = gr.Textbox(
                        show_label=False, 
                        placeholder="Type your question here...", 
                        scale=5
                    )
                    submit = gr.Button("Send", scale=1)

            # Right sidebar for real-time stock data and sentiment analysis
            with gr.Column(scale=3):
                with gr.Tab("Profile"):
                    gr.Markdown("### Your Financial Profile")
                    profile_info = gr.JSON(label="Profile Data")  # Remove the value parameter
                    generate_report_btn = gr.Button("Generate Financial Report")
                    report_output = gr.Markdown(label="Personal Financial Report")

                with gr.Tab("Market Data"):
                    stock_input = gr.Textbox(label="Enter Stock Ticker", placeholder="e.g. AAPL")
                    stock_btn = gr.Button("Get Stock Data")
                    stock_plot = gr.Plot(label="Stock Price Chart")
                    stock_info = gr.JSON(label="Stock Information")

                with gr.Tab("News Sentiment"):
                    news_input = gr.Textbox(label="Topic or Ticker", placeholder="e.g. bitcoin or TSLA")
                    news_btn = gr.Button("Analyze News Sentiment")
                    news_output = gr.JSON(label="News Sentiment Analysis")

                with gr.Tab("Market Mood"):
                    gr.Markdown("### Market Sentiment Tracker")
                    market_mood = gr.Label(label="Current Market Mood")
                    update_mood_btn = gr.Button("Update Market Mood")

        # Stock chart display area
        stock_chart_output = gr.Plot(visible=False)

        def user_input(message, chat_history, conversation_state, user_profile):
            if message == "":
                return chat_history, conversation_state, user_profile, None, gr.update(visible=False)

            chat_history, conversation_state, user_profile, chart = respond(message, chat_history, conversation_state, user_profile)

            # Return the updated user_profile instead of trying to update the JSON component directly
            if chart:
                return chat_history, conversation_state, user_profile, gr.update(value=""), gr.update(value=chart, visible=True)
            else:
                return chat_history, conversation_state, user_profile, gr.update(value=""), gr.update(visible=False)

        def initialize_chat():
            initial_message = "Hello! I'm your financial markets tutor. Before we begin, could you tell me your current level of knowledge about financial markets? Are you a beginner, intermediate, or advanced?"
            return [(None, initial_message)], [{"role": "assistant", "content": initial_message}], {}

        message.submit(user_input, [message, chatbot, conversation_state, user_profile], 
                       [chatbot, conversation_state, user_profile, message, stock_chart_output])
        submit.click(user_input, [message, chatbot, conversation_state, user_profile], 
                     [chatbot, conversation_state, user_profile, message, stock_chart_output])


        
        def fetch_stock_data(ticker):
            if not ticker or ticker.strip() == "":
                return None, "Please enter a valid ticker symbol"
            
            """Fetch real stock data using the existing get_stock_data function"""
            try:
                result = get_stock_data(ticker, period='3mo')
                if isinstance(result, str):
                    return None, result
            
                # Extract the chart and info from the result
                chart = result.get('chart', None)
                
                # Create a simplified info object with the most relevant data
                info = {
                    "ticker": result['ticker'],
                    "current_price": f"${result['current_price']:.2f}",
                    "weekly_change": f"{result['weekly_change']:.2f}%",
                    "monthly_change": f"{result['monthly_change']:.2f}%",
                    "volatility": f"{result['volatility']:.2f}%",
                    "above_SMA20": "Yes" if result['above_sma20'] else "No",
                    "above_SMA50": "Yes" if result['above_sma50'] else "No"
                }
                
                return chart, info
            except Exception as e:
                print(f"Error in fetch_stock_data: {str(e)}")
                return None, f"Error analyzing {ticker}: {str(e)}"
            
        def fetch_news_sentiment(topic):
            # Implementation to fetch and analyze news sentiment
            # Placeholder implementation:
            return {
                "topic": topic,
                "sentiment": round(np.random.normal(0, 1), 2),
                "articles": [
                    {"title": f"News about {topic} 1", "sentiment": "positive"},
                    {"title": f"News about {topic} 2", "sentiment": "neutral"},
                    {"title": f"News about {topic} 3", "sentiment": "negative"}
                ],
                "summary": f"Mixed sentiment around {topic} with slight positive bias."
            }
            
        def update_market_mood():
            # Implementation to update market mood
            moods = ["Bullish", "Bearish", "Neutral", "Fearful", "Greedy"]
            confidences = np.random.random(len(moods))
            confidences = confidences / confidences.sum()
            return {label: float(conf) for label, conf in zip(moods, confidences)}
            
        def generate_report(profile):
            # Implementation to generate financial report based on user profile
            if not profile:
                return "Please provide more information about your financial situation first."
                
            report = f"# Financial Report for {profile.get('name', 'User')}\n\n"
            report += "## Summary\n"
            report += "Based on your profile information, here's a personalized financial analysis.\n\n"
            
            # Add more sections based on available profile data
            if 'risk_tolerance' in profile:
                report += f"## Risk Assessment\n"
                report += f"Your risk tolerance is {profile['risk_tolerance']}.\n\n"
                
            return report
        
        # Update user profile based on message content
        # This is very basic - you would want more sophisticated parsing
        def user_input(message, chatbot, conversation_state, user_profile):
            chatbot.append((message, None))
    
            if "beginner" in message.lower():
                user_profile["knowledge_level"] = "beginner"
            elif "intermediate" in message.lower():
                user_profile["knowledge_level"] = "intermediate"
            elif "advanced" in message.lower():
                user_profile["knowledge_level"] = "advanced"
            
            response = "Got it! I'll tailor my answers to your knowledge level."
            conversation_state.append({"role": "assistant", "content": response})
            chatbot[-1] = (chatbot[-1][0], response)
    
            return chatbot, conversation_state, user_profile, "", None
        
        # Function to update the profile_info display
        def update_profile_display(user_profile):
            return user_profile
        
        # Stock data tab functionality
        stock_btn.click(fetch_stock_data, [stock_input], [stock_plot, stock_info])
        
        # News sentiment tab functionality
        news_btn.click(fetch_news_sentiment, [news_input], [news_output])
        
        # Market mood tab functionality
        update_mood_btn.click(update_market_mood, [], [market_mood])
        
        # Report generation functionality
        generate_report_btn.click(generate_report, [user_profile], [report_output])
        
        # Initialize the chat on page load
        demo.load(initialize_chat, [], [chatbot, conversation_state, user_profile])
        
        # Add a function to update the profile info whenever user_profile changes
        user_profile.change(update_profile_display, [user_profile], [profile_info])
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
