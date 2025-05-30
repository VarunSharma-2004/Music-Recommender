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
CLIENT_ID = "370457012435-s3efe495mtvr28ngbla812doqe5ca6ie.apps.googleusercontent.com" # YOUR NEW CLIENT ID
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

def load_chat_history(chat_file_path):
    if os.path.exists(chat_file_path):
        try:
            with open(chat_file_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {chat_file_path}. Returning empty history.")
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
    
    prompt_text = (
        "Based on the following conversation snippets between a user and a music chatbot, "
        "suggest a very short, unique, and catchy session name (max 4 words, ideally 2-3 words). "
        "Focus on key music themes, artists, or genres discussed. Avoid punctuation. Examples: 'Rock Anthems', 'Chill Vibes', 'Jazz Journey'.\n\n"
        "Conversation Snippets:\n"
    )
    relevant_messages = [msg for msg in messages_history if msg["role"] in ["user", "assistant"]][-6:]
    for msg in relevant_messages:
        prompt_text += f"{msg['role'].capitalize()}: {msg['content'][:100]}\n"
    
    try:
        name_model = genai.GenerativeModel("gemini-1.0-pro")
        response = name_model.generate_content(prompt_text)
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
    if sentiment_category == "positive":
        return ["Happy - Pharrell Williams", "Can't Stop the Feeling - Justin Timberlake", "Levitating - Dua Lipa", "Good as Hell - Lizzo", "Walking on Sunshine - Katrina & The Waves"]
    elif sentiment_category == "negative":
        return ["Someone Like You - Adele", "Let Her Go - Passenger", "Fix You - Coldplay", "Hallelujah - Leonard Cohen", "Everybody Hurts - R.E.M."]
    else:
        return ["Shape of You - Ed Sheeran", "Perfect - Ed Sheeran", "Wonderwall - Oasis", "Bohemian Rhapsody - Queen"]


# --- Streamlit UI and Logic ---
st.set_page_config(page_title="Music Chatbot", page_icon="🎶", layout="centered")

# Initialize session state variables
for key in ["user", "messages", "chat_file", "gemini_chat", "session_named_by_ai"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []
        if key == "session_named_by_ai":
            st.session_state[key] = False


# --- Google Login Flow ---
st.markdown("<h1 style='text-align: center;'>🎵 Music Chatbot 🎵</h1>", unsafe_allow_html=True)
query_params = st.query_params # st.experimental_get_query_params() is deprecated

if not st.session_state.user:
    if "code" not in query_params:
        st.markdown("### 🔗 Please login with Google to chat:")
        auth_url_params = {
            "client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI,
            "response_type": "code", "scope": SCOPE,
            "access_type": "offline", "prompt": "consent"
        }
        auth_url_full = f"{AUTH_BASE_URL}?{urlencode(auth_url_params)}"
        
        # DEBUGGING: Show the generated URL in the UI
        st.text_area("Debug: Copy this Auth URL and try it in a new browser tab if the button doesn't work as expected:", auth_url_full, height=100)
        print(f"DEBUG LOGIN URL: {auth_url_full}") # Also print to console

        login_button_html = f'''
        <a href="{auth_url_full}" target="_blank">
            <button style="display: block; margin: 10px auto; padding:10px 20px;font-size:16px;background-color:#4285F4;color:white;border:none;border-radius:5px;cursor:pointer;">
                👉 Login with Google
            </button>
        </a>'''
        st.markdown(login_button_html, unsafe_allow_html=True)
        st.stop()
    else: # "code" is in query_params, exchange for token
        code = query_params["code"][0] if isinstance(query_params["code"], list) else query_params["code"]
        token_data = {
            "code": code, "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code"
        }
        try:
            token_response = requests.post(TOKEN_URL, data=token_data)
            token_response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            tokens = token_response.json()
            access_token = tokens.get("access_token")

            if not access_token:
                st.error(f"Login failed: Access token not found in response. Response: {tokens}")
                st.stop()

            user_info_response = requests.get(USER_INFO_URL, headers={"Authorization": f"Bearer {access_token}"})
            user_info_response.raise_for_status()
            st.session_state.user = user_info_response.json()
            
            st.query_params.clear() # Clear query params
            st.rerun()

        except requests.exceptions.RequestException as e:
            error_detail = "No response details available."
            if e.response is not None:
                try:
                    error_detail = e.response.json() # Try to get JSON error from Google
                except json.JSONDecodeError:
                    error_detail = e.response.text # Fallback to text
            st.error(f"Login failed during token exchange or user info fetch: {e}. Details: {error_detail}")
            st.stop()

# --- Logged-in User View ---
# This part will only run if st.session_state.user is populated
user_data = st.session_state.user
user_name = user_data.get("name", "User")
user_email = user_data.get("email")

if not user_email: # Should not happen if login was successful
    st.error("Could not retrieve user email after login. Please try logging in again.")
    # Clear potentially corrupted session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

user_folder = os.path.join(CHAT_DIR, user_email.replace("@", "_at_").replace(".", "_"))
os.makedirs(user_folder, exist_ok=True)

# --- Chat Session Management (Sidebar) ---
with st.sidebar:
    st.markdown(f"👤 **{user_name}**")
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.query_params.clear()
        st.rerun()

    st.markdown("---")
    if st.button("➕ New Chat", use_container_width=True):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        new_chat_file = os.path.join(user_folder, f"Chat_{timestamp}.json")
        st.session_state.chat_file = new_chat_file
        st.session_state.messages = []
        st.session_state.gemini_chat = None
        st.session_state.session_named_by_ai = False
        save_chat_history(new_chat_file, [])
        st.rerun()

    st.markdown("📂 **Chat Sessions:**")
    try:
        session_files_list = sorted(
            [f for f in os.listdir(user_folder) if f.endswith(".json")],
            key=lambda f_name: os.path.getmtime(os.path.join(user_folder, f_name)),
            reverse=True
        )
    except FileNotFoundError:
        session_files_list = [] # User folder might not exist if it's a brand new user with no chats

    for file_name_ext in session_files_list:
        session_display_name = os.path.splitext(file_name_ext)[0]
        if st.button(session_display_name, key=f"load_{file_name_ext}", use_container_width=True):
            selected_path = os.path.join(user_folder, file_name_ext)
            st.session_state.chat_file = selected_path
            st.session_state.messages = load_chat_history(selected_path)
            st.session_state.gemini_chat = None
            st.session_state.session_named_by_ai = True # Assume loaded chats are already named
            st.rerun()

# --- Main Chat Interface ---
if not st.session_state.chat_file:
    if session_files_list: # Check if list is populated
        st.session_state.chat_file = os.path.join(user_folder, session_files_list[0])
        st.session_state.messages = load_chat_history(st.session_state.chat_file)
        st.session_state.gemini_chat = None
        st.session_state.session_named_by_ai = True
        st.rerun()
    else:
        # If no chat files, and "New Chat" wasn't clicked, prompt or create one
        # For simplicity, let's assume "New Chat" button will be used or the user selects one.
        # Or, to ensure a chat is always active after login if none exist:
        if st.session_state.user and not session_files_list: # Only if logged in and no chats
             timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
             new_chat_file = os.path.join(user_folder, f"Chat_{timestamp}.json")
             st.session_state.chat_file = new_chat_file
             st.session_state.messages = []
             st.session_state.gemini_chat = None
             st.session_state.session_named_by_ai = False
             save_chat_history(new_chat_file, [])
             st.rerun() # Rerun to load this new chat
        else:
            st.info("Select a chat session from the sidebar or start a '➕ New Chat'.")
            st.stop() # Stop execution if no chat is active and user isn't prompted to make one automatically


# Initialize Gemini Chat Model and Session
if st.session_state.chat_file and not st.session_state.gemini_chat:
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            system_instruction=SYSTEM_PROMPT_TEXT
        )
        gemini_history = []
        for msg in st.session_state.messages:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        st.session_state.gemini_chat = model.start_chat(history=gemini_history)
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {e}")
        # Consider st.stop() here if Gemini is essential for proceeding
        
if st.session_state.chat_file:
    st.subheader(f"Chatting: {os.path.basename(st.session_state.chat_file).replace('.json', '')}")
else: # Should ideally not be reached if logic above is correct
    st.subheader("Music Chatbot")
    st.info("Please select or start a new chat.")
    st.stop()


# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_prompt := st.chat_input("What music are you in the mood for?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if st.session_state.gemini_chat:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            thinking_text = "🎵 Thinking"
            for i in range(4): # Simple ... animation
                placeholder.markdown(thinking_text + "." * i)
                time.sleep(0.2)
            
            try:
                response = st.session_state.gemini_chat.send_message(user_prompt)
                bot_reply = response.text
            except ResourceExhausted as e:
                bot_reply = f"⚠️ API Quota Exceeded. Please try again later. ({e.message})"
                st.error(bot_reply)
            except Exception as e:
                bot_reply = f"😥 Sorry, an error occurred with Gemini: {e}"
                st.error(bot_reply)
            
            placeholder.markdown(bot_reply)

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        
        user_sentiment = analyze_sentiment(user_prompt)
        mood_keywords = ["sad", "happy", "energetic", "calm", "relax", "depressed", "excited", "bored", "gloomy", "cheerful", "upbeat"]
        
        if user_sentiment != "neutral" or any(keyword in user_prompt.lower() for keyword in mood_keywords) :
            with st.chat_message("assistant"):
                spotify_placeholder = st.empty()
                spotify_placeholder.markdown("🎶 Let me find some tunes for that mood...")
                time.sleep(0.5)

                recommended_song_names = recommend_songs_for_sentiment(user_sentiment)
                spotify_tracks_info = search_spotify_tracks(recommended_song_names)

                if spotify_tracks_info:
                    spotify_reply = f"Based on your mood ({user_sentiment}), you might like these:\n"
                    for track in spotify_tracks_info:
                        spotify_reply += f"\n- **{track['Song']}** by *{track['Artist']}* ([Listen on Spotify]({track['Link']}))"
                    spotify_placeholder.markdown(spotify_reply)
                    st.session_state.messages.append({"role": "assistant", "content": spotify_reply})
                else:
                    spotify_placeholder.markdown("I couldn't find specific Spotify tracks for that mood right now.")
        
        save_chat_history(st.session_state.chat_file, st.session_state.messages)

        if len(st.session_state.messages) >= 4 and not st.session_state.get("session_named_by_ai", False):
            # ... (session naming logic - kept as is for brevity, ensure it's robust) ...
            new_session_name_str = generate_session_name_from_gemini(st.session_state.messages)
            if new_session_name_str:
                safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in new_session_name_str).strip()
                if not safe_name or safe_name.lower().startswith("chat_"):
                    safe_name = f"MusicSession_{time.strftime('%H%M')}"

                new_file_path = os.path.join(user_folder, f"{safe_name}.json")
                current_file_path = st.session_state.chat_file
                
                if current_file_path != new_file_path and os.path.exists(current_file_path):
                    counter = 1
                    temp_new_path = new_file_path
                    while os.path.exists(temp_new_path) and temp_new_path != current_file_path : # prevent renaming to itself if collision happens with base name
                        temp_new_path = os.path.join(user_folder, f"{safe_name}_{counter}.json")
                        counter += 1
                    new_file_path = temp_new_path

                    if new_file_path != current_file_path : # Final check
                        try:
                            os.rename(current_file_path, new_file_path)
                            st.session_state.chat_file = new_file_path
                            st.session_state.session_named_by_ai = True
                            st.rerun()
                        except OSError as e:
                            print(f"Error renaming chat file from {current_file_path} to {new_file_path}: {e}")
                elif not os.path.exists(current_file_path): # Edge case: current file was deleted somehow
                     st.session_state.chat_file = new_file_path
                     save_chat_history(new_file_path, st.session_state.messages)
                     st.session_state.session_named_by_ai = True
                     st.rerun()


    else:
        st.error("Chat session with Gemini not properly initialized. Please try creating a new chat or reloading.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>🎧 Happy Chatting! 🎧</p>", unsafe_allow_html=True)
