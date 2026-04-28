import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def search_web(query: str) -> str:
    """
    Search the web using Tavily to find real-time information about
    specific businesses, restaurants, attractions, hotels, and travel details.
    Use this whenever you need current, grounded data about a destination.

    Args:
        query (str): The search query, e.g. "best ramen restaurants Shibuya Tokyo 2025"

    Returns:
        str: Search results with real business names, addresses, and details
    """
    try:
        if not query or not query.strip():
            return "Error: Missing search query"

        # ================== TAVILY SEARCH TOOL ==================
        tavily = TavilySearch(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            topic="general",
        )
        results = tavily.invoke({"query": query})

        if not results:
            return "No results found for that query."

        # Format results into clean readable text for the LLM
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content available")
            formatted.append(f"[{i}] {title}\nURL: {url}\n{content}\n")

        return "\n".join(formatted)

    except Exception as e:
        logger.error(f"Error in search_web: {str(e)}")
        return f"Error: Tavily search failed - {str(e)}"


@tool
def send_itinerary_email(
    recipient_email: str,
    subject: str,
    itinerary_content: str
) -> str:
    """
    Send the trip itinerary via email using Gmail SMTP.

    Args:
        recipient_email (str): The email address of the recipient
        subject (str): The email subject
        itinerary_content (str): The itinerary content to send

    Returns:
        str: Success or error message
    """
    try:
        if not recipient_email:
            return "Error: Missing recipient email address"
        if not subject:
            return "Error: Missing email subject"
        if not itinerary_content or not itinerary_content.strip():
            return "Error: No itinerary content to send"
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', recipient_email):
            return "Error: Invalid email format"

        sender_email = os.getenv("GMAIL_USER")
        sender_password = os.getenv("GMAIL_APP_PASS")

        if not sender_email or not sender_password:
            return "Error: Gmail credentials not configured. Please set GMAIL_USER and GMAIL_APP_PASSWORD environment variables."

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = recipient_email

        html_content = f"""
        <html>
        <body>
            <h2>{subject}</h2>
            <pre>{itinerary_content}</pre>
            <p><small>This email was sent automatically by VoyageGT travel assistant.</small></p>
        </body>
        </html>
        """

        part = MIMEText(html_content, "html")
        message.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)

        logger.info(f"Successfully sent itinerary to {recipient_email}")
        return f"Successfully sent itinerary to {recipient_email}"

    except smtplib.SMTPAuthenticationError:
        return "Error: Gmail authentication failed. Check your credentials."
    except smtplib.SMTPRecipientsRefused:
        return "Error: Email address refused by SMTP server."
    except smtplib.SMTPServerDisconnected:
        return "Error: Connection to email server failed. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error in send_itinerary_email: {str(e)}")
        return f"Error: Failed to send email - {str(e)}"


@tool
def generate_itinerary(
    destination: str,
    duration: int,
    interests: str,
    budget: str,
    travel_dates: str
) -> str:
    """
    Generate a detailed travel itinerary for a destination.

    Args:
        destination (str): The travel destination
        duration (int): Number of days for the trip
        interests (str): User's interests (e.g., museums, food, nature)
        budget (str): Budget level (e.g., low, medium, high)
        travel_dates (str): Travel dates in any format the user provided,
            e.g. "September 2026", "June 15-30 2026", "late October" —
            do not require or request a specific date format

    Returns:
        str: Generated itinerary
    """
    try:
        if not destination or not destination.strip():
            return "Error: Missing destination"
        if not isinstance(duration, int) or duration <= 0:
            return "Error: Invalid duration"
        if not interests or not interests.strip():
            interests = "general sightseeing"
        if not budget or not budget.strip():
            budget = "medium"

        itinerary = f"""
Travel Itinerary for {destination}
Duration: {duration} days
Interests: {interests}
Budget: {budget}
Travel Dates: {travel_dates}

Day 1: Arrival and welcome
- Arrive at the airport
- Check-in at hotel
- Evening stroll around the city center

Day 2: Cultural exploration
- Visit {destination}'s main museum
- Explore local markets
- Dinner at recommended restaurant

Day 3: Nature and outdoor activities
- Visit {destination}'s natural attractions
- Local hiking trail
- Evening cultural show

Day 4: Food and local experience
- Cooking class
- Local food tour
- Dinner with locals

Day 5: Departure
- Last-minute shopping
- Check-out and transfer to airport

Total estimated cost: $500-1000 (excluding flights)
"""
        logger.info(f"Generated itinerary for {destination}")
        return itinerary

    except Exception as e:
        logger.error(f"Error in generate_itinerary: {str(e)}")
        return f"Error: Failed to generate itinerary - {str(e)}"


# ================== LIST OF ALL TOOLS ==================
tools = [
    search_web,
    send_itinerary_email,
    generate_itinerary,
]


# Test the tools
if __name__ == "__main__":
    # Test send_itinerary_email (uncomment to test - requires email setup)
    # result = send_itinerary_email("test@example.com", "Test Subject", "Test content")
    # print("Email tool result:", result)

    # Test generate_itinerary
    result = generate_itinerary(
        destination="Paris",
        duration=5,
        interests="museums, food, art",
        budget="high",
        travel_dates="2023-06-01 to 2023-06-05"
    )
    print("Itinerary tool result:", result)