import streamlit as st
import requests
from urllib.parse import urlencode
import os
import json
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted # Import the specific exception
from textblob import TextBlob
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

# --- Constants and Configurations ---
# Google Gemini API Key
GEMINI_API_KEY = os.getenv("gemini_api")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set it in your .env file.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# Google OAuth Setup
CLIENT_ID = "365256296387-8qe3cfm2lmj46jk832arf0ltevfpqg93.apps.googleusercontent.com"
CLIENT_SECRET = os.getenv("client_secret")
if not CLIENT_SECRET:
    st.error("Google Client Secret not found. Please set it in your .env file.")
    st.stop()
REDIRECT_URI = "https://music-gpt.streamlit.app" # Ensure this matches your Google Cloud Console
SCOPE = "openid email profile"
AUTH_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USER_INFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("spotify_client_id")
SPOTIFY_CLIENT_SECRET = os.getenv("spotify_client_secret")
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    st.error("Spotify client ID or secret not found. Please set them in your .env file.")
    st.stop()
try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))
except Exception as e:
    st.error(f"Failed to initialize Spotify client: {e}")
    st.stop()


CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)

# System prompt for Gemini
SYSTEM_PROMPT_TEXT = (
    "You are a helpful AI chatbot specialized in music. "
    "You can recommend songs, discuss genres, suggest playlists based on mood, and talk about artists. "
    "If the user asks anything unrelated to music (like weather, sports, politics, or general chit-chat), "
    "reply: 'Sorry, I'm a music chatbot. I can only help you with music recommendations and related discussions.' "
    "When recommending songs, try to mention the artist as well."
)

# --- Helper Functions ---
def search_spotify_tracks(song_names):
    results = []
    if not song_names:
        return results
    for name in song_names:
        try:
            result = sp.search(q=name, type='track', limit=1)
            if result and result["tracks"]["items"]:
                track = result["tracks"]["items"][0]
                results.append({
                    "Song": track["name"],
                    "Artist": ", ".join([a["name"] for a in track["artists"]]),
                    "Link": track["external_urls"]["spotify"]
                })
        except Exception as e:
            print(f"Error searching Spotify for '{name}': {e}") # Log error
    return results

# This function seems redundant if you're using the spotipy library's auth_manager
# Spotipy handles token refreshing automatically.
# def get_spotify_token():
#     auth_response = requests.post(
#         "https://accounts.spotify.com/api/token",
#         data={"grant_type": "client_credentials"},
#         auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
#     )
#     auth_response.raise_for_status() # Raise an exception for bad status codes
#     return auth_response.json().get("access_token")

def load_chat_history(chat_file_path):
    if os.path.exists(chat_file_path):
        try:
            with open(chat_file_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return [] # Return empty if file is corrupted
    return []

def save_chat_history(chat_file_path, chat_history_list):
    try:
        with open(chat_file_path, "w") as file:
            json.dump(chat_history_list, file, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")


def generate_session_name_from_gemini(messages_history):
    if not messages_history:
        return f"Chat_{int(time.time())}"
    
    prompt = (
        "Based on the following conversation snippets between a user and a music chatbot, "
        "suggest a very short, unique, and catchy session name (max 4 words, ideally 2-3 words). "
        "Focus on key music themes, artists, or genres discussed. Avoid punctuation. Examples: 'Rock Anthems', 'Chill Vibes', 'Jazz Journey'.\n\n"
        "Conversation Snippets:\n"
    )
    # Use only a few relevant messages to keep the prompt short
    relevant_messages = [msg for msg in messages_history if msg["role"] in ["user", "assistant"]][-6:] # Last 6 messages
    for msg in relevant_messages:
        prompt += f"{msg['role'].capitalize()}: {msg['content'][:100]}\n" # Truncate long messages
    
    try:
        # Use a more cost-effective model if possible for summarization, like gemini-1.0-pro if sufficient,
        # or stick to 1.5-pro if complex understanding is needed.
        # For naming, a simpler model might be fine.
        name_model = genai.GenerativeModel("gemini-1.0-pro") # Or gemini-1.5-flash when available
        response = name_model.generate_content(prompt)
        name = response.text.strip().replace("\n", " ").replace(":", "").replace('"', "").title()
        return name if name else f"Chat_{int(time.time())}"
    except Exception as e:
        print(f"Error generating session name: {e}")
        return f"Chat_{int(time.time())}"

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.2: return "positive"
    if polarity < -0.2: return "negative"
    return "neutral"

def recommend_songs_for_sentiment(sentiment_category):
    # This is your predefined list. Consider making it more dynamic or larger.
    if sentiment_category == "positive":
        return ["Happy - Pharrell Williams", "Can't Stop the Feeling - Justin Timberlake", "Levitating - Dua Lipa", "Good as Hell - Lizzo", "Walking on Sunshine - Katrina & The Waves"]
    elif sentiment_category == "negative": # 'sad' or 'negative'
        return ["Someone Like You - Adele", "Let Her Go - Passenger", "Fix You - Coldplay", "Hallelujah - Leonard Cohen", "Everybody Hurts - R.E.M."]
    # Add more moods if needed
    # elif sentiment_category == "energetic":
    #     return ["Titanium - David Guetta", "Blinding Lights - The Weeknd", "Lose Yourself - Eminem"]
    else: # neutral or other unrecognized sentiments
        return ["Shape of You - Ed Sheeran", "Perfect - Ed Sheeran", "Wonderwall - Oasis", "Bohemian Rhapsody - Queen"]


# --- Streamlit UI and Logic ---
st.set_page_config(page_title="Music Chatbot", page_icon="🎶", layout="centered")

# Initialize session state variables if they don't exist
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_file" not in st.session_state:
    st.session_state.chat_file = None
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = None
if "session_named_by_ai" not in st.session_state:
    st.session_state.session_named_by_ai = False # Flag to ensure renaming happens once per session length threshold

# --- Google Login Flow ---
st.markdown("<h1 style='text-align: center;'>🎵 Music Chatbot 🎵</h1>", unsafe_allow_html=True)
query_params = st.query_params

if not st.session_state.user:
    if "code" not in query_params:
        st.markdown("### 🔗 Please login with Google to chat:")
        auth_url_params = {
            "client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI,
            "response_type": "code", "scope": SCOPE,
            "access_type": "offline", "prompt": "consent"
        }
        auth_url_full = f"{AUTH_BASE_URL}?{urlencode(auth_url_params)}"
        login_button_html = f'''
        <a href="{auth_url_full}" target="_self">
            <button style="display: block; margin: 10px auto; padding:10px 20px;font-size:16px;background-color:#4285F4;color:white;border:none;border-radius:5px;cursor:pointer;">
                👉 Login with Google
            </button>
        </a>'''
        st.markdown(login_button_html, unsafe_allow_html=True)
        st.stop()
    else: # "code" is in query_params, exchange for token
        code = query_params["code"]
        token_data = {
            "code": code, "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code"
        }
        try:
            token_response = requests.post(TOKEN_URL, data=token_data)
            token_response.raise_for_status()
            tokens = token_response.json()
            access_token = tokens.get("access_token")

            user_info_response = requests.get(USER_INFO_URL, headers={"Authorization": f"Bearer {access_token}"})
            user_info_response.raise_for_status()
            st.session_state.user = user_info_response.json()
            
            # Clear query params and rerun to remove code from URL and proceed to chat
            st.query_params.clear() # Use st.query_params.clear()
            st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Login failed: {e}. Details: {e.response.text if e.response else 'No response details'}")
            st.stop()

# --- Logged-in User View ---
user_data = st.session_state.user
user_name = user_data.get("name", "User")
user_email = user_data.get("email")

if not user_email:
    st.error("Could not retrieve user email. Please try logging in again.")
    st.session_state.clear()
    st.rerun()

user_folder = os.path.join(CHAT_DIR, user_email.replace("@", "_at_").replace(".", "_"))
os.makedirs(user_folder, exist_ok=True)

# --- Chat Session Management (Sidebar) ---
with st.sidebar:
    st.markdown(f"👤 **{user_name}**")
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()): # Iterate over a copy of keys
            del st.session_state[key]
        st.query_params.clear()
        st.rerun()

    st.markdown("---")
    if st.button("➕ New Chat", use_container_width=True):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        new_chat_file = os.path.join(user_folder, f"Chat_{timestamp}.json") # Generic name initially
        st.session_state.chat_file = new_chat_file
        st.session_state.messages = []
        st.session_state.gemini_chat = None # Crucial: Reset Gemini chat object for new session
        st.session_state.session_named_by_ai = False # Reset naming flag
        save_chat_history(new_chat_file, [])
        st.rerun()

    st.markdown("📂 **Chat Sessions:**")
    session_files = sorted(
        [f for f in os.listdir(user_folder) if f.endswith(".json")],
        key=lambda f: os.path.getmtime(os.path.join(user_folder, f)),
        reverse=True
    )

    for file_name in session_files:
        session_display_name = os.path.splitext(file_name)[0] # Remove .json
        
        # Use columns for better layout if you add rename/delete again
        # For now, just a button to load
        if st.button(session_display_name, key=f"load_{file_name}", use_container_width=True):
            selected_path = os.path.join(user_folder, file_name)
            st.session_state.chat_file = selected_path
            st.session_state.messages = load_chat_history(selected_path)
            st.session_state.gemini_chat = None # Reset, will be re-initialized
            st.session_state.session_named_by_ai = True # Assume loaded chats are already named or don't need immediate auto-rename
            st.rerun()
        
        # Consider adding delete/download later if needed, keeping it simple for now

# --- Main Chat Interface ---
if not st.session_state.chat_file: # If no chat is selected (e.g., after login, before "New Chat")
    st.info("Select a chat session from the sidebar or start a 'New Chat'.")
    # Optionally, auto-select the latest or start a new one:
    if session_files:
        st.session_state.chat_file = os.path.join(user_folder, session_files[0])
        st.session_state.messages = load_chat_history(st.session_state.chat_file)
        st.session_state.gemini_chat = None
        st.session_state.session_named_by_ai = True
        st.rerun()
    else: # No existing chats, force new chat creation
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        new_chat_file = os.path.join(user_folder, f"Chat_{timestamp}.json")
        st.session_state.chat_file = new_chat_file
        st.session_state.messages = []
        st.session_state.gemini_chat = None
        st.session_state.session_named_by_ai = False
        save_chat_history(new_chat_file, [])
        st.rerun()


# Initialize Gemini Chat Model and Session if not already done for the current chat_file
if st.session_state.chat_file and not st.session_state.gemini_chat:
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest", # Or your preferred model
            system_instruction=SYSTEM_PROMPT_TEXT
        )
        
        # Rebuild history for Gemini from st.session_state.messages
        gemini_history = []
        for msg in st.session_state.messages:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        
        st.session_state.gemini_chat = model.start_chat(history=gemini_history)
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {e}")
        st.stop()


st.subheader(f"Chatting: {os.path.basename(st.session_state.chat_file).replace('.json', '')}")

# Display chat messages from Streamlit's session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_prompt := st.chat_input("What music are you in the mood for?"):
    # Append user message to Streamlit's message list and display
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get response from Gemini
    if st.session_state.gemini_chat:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            thinking_text = "🎵 Thinking..."
            for i in range(len(thinking_text) + 1):
                placeholder.markdown(thinking_text[:i] + ("..." if i < len(thinking_text) else ""))
                time.sleep(0.05)
            
            try:
                response = st.session_state.gemini_chat.send_message(user_prompt)
                bot_reply = response.text
            except ResourceExhausted as e:
                bot_reply = f"⚠️ I'm a bit overwhelmed with requests right now. Please try again in a minute. (Details: Quota Exceeded)"
                st.error(bot_reply)
            except Exception as e:
                bot_reply = f"😥 Sorry, an error occurred: {e}"
                st.error(bot_reply)
            
            placeholder.markdown(bot_reply)

        # Append bot response to Streamlit's message list
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        
        # Spotify song recommendation logic based on sentiment of user's last message
        # This happens *after* Gemini's response.
        user_sentiment = analyze_sentiment(user_prompt)
        mood_keywords = ["sad", "happy", "energetic", "calm", "relax", "depressed", "excited", "bored", "gloomy", "cheerful", "upbeat"]
        
        if user_sentiment != "neutral" or any(keyword in user_prompt.lower() for keyword in mood_keywords) :
            with st.chat_message("assistant"): # A follow-up message from assistant
                spotify_placeholder = st.empty()
                spotify_placeholder.markdown("🎶 Let me find some tunes for that mood...")
                time.sleep(1) # Brief pause

                recommended_song_names = recommend_songs_for_sentiment(user_sentiment)
                spotify_tracks_info = search_spotify_tracks(recommended_song_names)

                if spotify_tracks_info:
                    spotify_reply = f"Based on your mood ({user_sentiment}), you might like these:\n\n"
                    for track in spotify_tracks_info:
                        spotify_reply += f"- **{track['Song']}** by *{track['Artist']}* ([Listen on Spotify]({track['Link']}))\n"
                    spotify_placeholder.markdown(spotify_reply)
                    st.session_state.messages.append({"role": "assistant", "content": spotify_reply})
                else:
                    spotify_placeholder.markdown("I couldn't find specific Spotify tracks for that mood right now, but I hope my previous advice helps!")
        
        # Save history
        save_chat_history(st.session_state.chat_file, st.session_state.messages)

        # Auto-rename session file based on content (once)
        if len(st.session_state.messages) >= 4 and not st.session_state.get("session_named_by_ai", False):
            new_session_name = generate_session_name_from_gemini(st.session_state.messages)
            if new_session_name:
                safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in new_session_name).strip()
                if not safe_name or safe_name.lower().startswith("chat_"): # Ensure a decent name
                    safe_name = f"MusicSession_{time.strftime('%H%M')}"

                new_file_path = os.path.join(user_folder, f"{safe_name}.json")
                
                current_file_path = st.session_state.chat_file
                if current_file_path != new_file_path: # Only rename if different
                    if os.path.exists(current_file_path):
                         # Avoid overwriting if the generated name (unlikely) collides
                        counter = 1
                        temp_path = new_file_path
                        while os.path.exists(temp_path) and temp_path != current_file_path:
                            temp_path = os.path.join(user_folder, f"{safe_name}_{counter}.json")
                            counter += 1
                        new_file_path = temp_path
                        
                        if new_file_path != current_file_path:
                            try:
                                os.rename(current_file_path, new_file_path)
                                st.session_state.chat_file = new_file_path
                                st.session_state.session_named_by_ai = True # Mark as named
                                st.rerun() # Rerun to update sidebar with new name
                            except OSError as e:
                                print(f"Error renaming chat file: {e}")
                    else: # Current file somehow doesn't exist, just update session state
                        st.session_state.chat_file = new_file_path
                        save_chat_history(new_file_path, st.session_state.messages) # Save with new name
                        st.session_state.session_named_by_ai = True
                        st.rerun()


    else:
        st.error("Chat session not properly initialized. Please try creating a new chat.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>🎧 Happy Chatting! 🎧</p>", unsafe_allow_html=True)
