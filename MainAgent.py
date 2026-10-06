import os
import requests

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from tavily import TavilyClient


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing.")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing.")


# =========================================================
# 2. TAVILY CLIENT
# =========================================================

tavily_client = TavilyClient(
    api_key=tavily_api_key
)


# =========================================================
# 3. WEATHER TOOL
# =========================================================

@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a city.

    ALWAYS use this tool when the user asks about:
    - current weather
    - current temperature
    - rain
    - humidity
    - weather conditions
    - whether it is hot or cold

    Do not answer current weather questions from your own knowledge.
    Use this tool to get the current weather.
    """

    print(f"\n[TOOL CALL] get_weather({location})")

    # -----------------------------------------------------
    # STEP 1: Convert city name into latitude and longitude
    # -----------------------------------------------------

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

    geocoding_params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        geocoding_url,
        params=geocoding_params,
        timeout=10
    )

    if response.status_code != 200:
        return f"Could not find location: {location}"

    data = response.json()

    if "results" not in data:
        return f"Could not find location: {location}"

    place = data["results"][0]

    latitude = place["latitude"]
    longitude = place["longitude"]

    city = place["name"]
    country = place.get("country", "")


    # -----------------------------------------------------
    # STEP 2: Get current weather
    # -----------------------------------------------------

    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,weather_code",
        "timezone": "auto"
    }

    response = requests.get(
        weather_url,
        params=weather_params,
        timeout=10
    )

    if response.status_code != 200:
        return "Unable to get current weather."


    # -----------------------------------------------------
    # STEP 3: Read weather response
    # -----------------------------------------------------

    weather_data = response.json()

    current = weather_data["current"]

    temperature = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]
    weather_code = current["weather_code"]


    # -----------------------------------------------------
    # STEP 4: Convert weather code into simple description
    # -----------------------------------------------------

    weather_description = get_weather_description(weather_code)


    # -----------------------------------------------------
    # STEP 5: Create result for the AI agent
    # -----------------------------------------------------

    result = (
        f"Current weather in {city}, {country}: "
        f"{weather_description}. "
        f"Temperature: {temperature}°C. "
        f"Humidity: {humidity}%."
    )

    print("[TOOL RESULT]", result)

    return result


# =========================================================
# WEATHER CODE HELPER
# =========================================================

def get_weather_description(code):

    weather_codes = {

        0: "Clear sky",

        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",

        45: "Fog",
        48: "Depositing rime fog",

        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",

        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",

        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",

        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",

        95: "Thunderstorm",

        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }

    return weather_codes.get(
        code,
        "Unknown weather condition"
    )


# =========================================================
# 4. ADD TOOL
# =========================================================

@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers.

    ALWAYS use this tool when the user asks
    to calculate the sum of two numbers.
    """

    print(f"\n[TOOL CALL] add({a}, {b})")

    result = a + b

    print("[TOOL RESULT]", result)

    return result


# =========================================================
# 5. SUBTRACT TOOL
# =========================================================

@tool
def subtract(a: int, b: int) -> int:
    """
    Subtract b from a.

    ALWAYS use this tool when the user asks
    to subtract one number from another.
    """

    print(f"\n[TOOL CALL] subtract({a}, {b})")

    result = a - b

    print("[TOOL RESULT]", result)

    return result


# =========================================================
# 6. WEB SEARCH TOOL
# =========================================================

@tool
def web_search(query: str) -> str:
    """
    Search the internet.

    ALWAYS use this tool when the user asks for:
    - latest information
    - current news
    - recent events
    - information that may have changed recently
    """

    print(f"\n[TOOL CALL] web_search({query})")

    response = tavily_client.search(
        query=query,
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return "No search results found."

    result = "\n".join(
        f"- {r['title']}: {r['content']}"
        for r in results
    )

    print("[TOOL RESULT]", result)

    return result


# =========================================================
# 7. REGISTER ALL TOOLS
# =========================================================

tools = [
    get_weather,
    add,
    subtract,
    web_search
]


# =========================================================
# 8. CREATE LLM
# =========================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="openai/gpt-oss-120b",
    temperature=0
)


# =========================================================
# 9. CREATE AI AGENT
# =========================================================

graph = create_react_agent(
    llm,
    tools
)


# =========================================================
# 10. MAIN CHAT LOOP
# =========================================================

if __name__ == "__main__":

    print("==========================================")
    print("       SIMPLE AI TOOL-USING AGENT")
    print("==========================================")

    print("""
Available tools:

1. Weather
2. Calculator
3. Web Search

Examples:

"What is the weather in Pune?"
"What is the temperature in London?"
"What is 25 + 75?"
"What is 100 - 37?"
"What are the latest developments in AI?"

Type 'exit' to quit.
""")

    conversation = []

    while True:

        user_input = input("\nYou: ")

        # Exit program
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Add user's message
        conversation.append({
            "role": "user",
            "content": user_input
        })

        # Send conversation to agent
        result = graph.invoke({
            "messages": conversation
        })

        # Store updated conversation
        conversation = result["messages"]

        # Print final AI response
        print("\nAI:", conversation[-1].content)