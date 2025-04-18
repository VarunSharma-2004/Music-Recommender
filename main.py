import streamlit as st
import requests
from urllib.parse import urlencode
import os
import json
import time
import google.generativeai as genai
from textblob import TextBlob
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
load_dotenv()

# Google Gemini API Key
genai.configure(api_key="AIzaSyAEc2vZc-_QbAz753Umil7d0Na7jlbblUI")

# Google OAuth Setup
client_id = "799230987280-slj2i30suidm19oa69k7uuf3fv8mghe1.apps.googleusercontent.com"
client_secret = os.getenv("client_secret")
redirect_uri = "https://music-gpt.streamlit.app"
scope = "openid email profile"
auth_base_url = "https://accounts.google.com/o/oauth2/v2/auth"
token_url = "https://oauth2.googleapis.com/token"
user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"

# Spotify API credentials
spotify_client_id = os.getenv("spotify_client_id")
spotify_client_secret = os.getenv("spotify_client_secret")
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=spotify_client_id, client_secret=spotify_client_secret))

def search_spotify_tracks(song_names):
    results = []
    for name in song_names:
        result = sp.search(q=name, type='track', limit=1)
        if result["tracks"]["items"]:
            track = result["tracks"]["items"][0]
            results.append({
                "Song": track["name"],
                "Artist": track["artists"][0]["name"],
                "Link": track["external_urls"]["spotify"]
            })
    return results

# Get Spotify access token
def get_spotify_token():
    auth_response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(spotify_client_id, spotify_client_secret)
    )
    return auth_response.json().get("access_token")


# File paths
CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)

# Auth URL
params = {
    "client_id": client_id,
    "redirect_uri": redirect_uri,
    "response_type": "code",
    "scope": scope,
    "access_type": "offline",
    "prompt": "consent"
}
auth_url = f"{auth_base_url}?{urlencode(params)}"

# Load/save functions
def load_chat_history(chat_file):
    if os.path.exists(chat_file):
        with open(chat_file, "r") as file:
            return json.load(file)
    return []

def save_chat_history(chat_file, chat_history):
    with open(chat_file, "w") as file:
        json.dump(chat_history, file)

# Streamlit UI
st.set_page_config(page_title="Music Chatbot", page_icon="🎶", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎵 Music Chatbot Login 🎵</h1>", unsafe_allow_html=True)

# Google Login
query_params = st.query_params
if "code" not in query_params and "user" not in st.session_state:
    st.markdown("### 🔗 Please login with Google:")
    login_button_html = f"""
    <a href="{auth_url}" target="_blank">
        <button style="padding:10px 20px;font-size:16px;background-color:#4285F4;color:white;border:none;border-radius:5px;cursor:pointer;">
            👉 Login with Google
        </button>
    </a>
"""
    st.markdown(login_button_html, unsafe_allow_html=True)

    st.stop()

# Exchange token if code is present
if "code" in query_params and "user" not in st.session_state:
    code = query_params["code"]
    token_data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }

    token_response = requests.post(token_url, data=token_data)
    if token_response.status_code == 200:
        tokens = token_response.json()
        access_token = tokens.get("access_token")

        user_info = requests.get(
            user_info_url,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        if user_info.status_code == 200:
            st.session_state["user"] = user_info.json()
            st.session_state["messages"] = []
            st.query_params.clear()
            st.rerun()
        else:
            st.error("Failed to fetch user info")
            st.stop()
    else:
        st.error("Failed to exchange code")
        st.stop()

# Logged-in view
user = st.session_state["user"]
user_name = user.get("name")
user_email = user.get("email")

# User-specific chat folder
user_folder = os.path.join(CHAT_DIR, user_email.replace("@", "_at_"))
os.makedirs(user_folder, exist_ok=True)

# Create a new session on fresh login
if "chat_file" not in st.session_state:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    chat_file = os.path.join(user_folder, f"{timestamp}.json")
    st.session_state["chat_file"] = chat_file
    st.session_state["messages"] = []
else:
    chat_file = st.session_state["chat_file"]

# Sidebar with logout + recent searches
with st.sidebar:
    st.markdown(f"👤 Logged in as: **{user_name}**")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

    st.markdown("---")
    # ➕ New Chat Button
    if st.button("➕ New Chat"):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        new_chat_file = os.path.join(user_folder, f"{timestamp}.json")
        st.session_state["chat_file"] = new_chat_file
        st.session_state["messages"] = []
        st.session_state["named"] = False
        save_chat_history(new_chat_file, [])
        st.rerun()
    st.markdown("📂 **Chat Sessions:**")

    session_files = sorted(os.listdir(user_folder), reverse=True)

    for file in session_files:
        session_name = file.replace(".json", "")
        cols = st.columns([6, 1, 1, 1])  # For name, rename, delete, download

        with cols[0]:
            if st.button(session_name, key=file):
                selected_path = os.path.join(user_folder, file)
                st.session_state["chat_file"] = selected_path
                st.session_state["messages"] = load_chat_history(selected_path)
                st.rerun()

        with cols[1]:
            if cols[1].button("🗑️", key=f"delete_{file}"):
                os.remove(os.path.join(user_folder, file))
                if st.session_state["chat_file"].endswith(file):
                    st.session_state["chat_file"] = None
                    st.session_state["messages"] = []
                st.warning(f"Deleted {session_name}")
                st.rerun()

        with cols[2]:
            if cols[2].download_button("⬇️", data=json.dumps(load_chat_history(os.path.join(user_folder, file)), indent=2), file_name=file, mime="application/json", key=f"download_{file}"):
                pass

# Initialize chat
if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat_history(chat_file)

st.subheader("Start chatting with the Music Bot 🎤")

# Display chat
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def generate_session_name(messages):
    prompt = (
        "Based on the following conversation between a user and a music chatbot, "
        "suggest a short and catchy session name (max 5 words). Avoid punctuation and keep it unique.\n\n"
    )
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    try:
        response = model.generate_content(prompt)
        name = response.text.strip().replace("\n", "")
        return name if name else f"Untitled_{int(time.time())}"
    except:
        return f"Untitled_{int(time.time())}"

# Sentiment Analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
    
    if sentiment > 0.2:
        return "positive"
    elif sentiment < -0.2:
        return "negative"
    else:
        return "neutral"

# Spotify music recommendation based on mood
def recommend_music_based_on_sentiment(sentiment):
    access_token = get_spotify_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    if sentiment == "positive":
        query = "party upbeat"
    elif sentiment == "negative":
        query = "sad chill"
    else:
        query = "relax calm"

    params = {
        "q": query,
        "type": "track",
        "limit": 3,
        "market": "IN"
    }
    response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)

    if response.status_code == 200:
        results = response.json()
        tracks = results.get("tracks", {}).get("items", [])
        recommendations = []
        for track in tracks:
            name = track["name"]
            artists = ", ".join([artist["name"] for artist in track["artists"]])
            url = track["external_urls"]["spotify"]
            recommendations.append(f"[{name} - {artists}]({url})")
        return "\n\n".join(recommendations)
    else:
        return "Sorry, I couldn't fetch music suggestions right now."
    
def recommend_music_list(sentiment):
    if sentiment == "happy":
        return ["Happy - Pharrell Williams", "Can't Stop the Feeling - Justin Timberlake", "Levitating - Dua Lipa"]
    elif sentiment == "energetic":
        return ["Titanium - David Guetta", "Blinding Lights - The Weeknd", "Lose Yourself - Eminem"]
    elif sentiment == "sad":
        return ["Someone Like You - Adele", "Let Her Go - Passenger", "Fix You - Coldplay"]
    else:
        return ["Shape of You - Ed Sheeran", "Perfect - Ed Sheeran"]

# Chat input
user_message = st.chat_input("Type your message...")
if user_message:
    st.session_state["messages"].append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        thinking_text = "Thinking..."
        for i in range(len(thinking_text)):
            time.sleep(0.03)
            placeholder.markdown(thinking_text[:i+1])

        # Store chat session once per user
        if "gemini_chat" not in st.session_state:
            model = genai.GenerativeModel("gemini-1.5-pro")

            system_prompt = (
                "You are a helpful AI chatbot specialized in music. "
                "You can recommend songs, discuss genres, suggest playlists based on mood, and talk about artists. "
                "If the user asks anything unrelated to music (like weather, sports, politics, or general chit-chat), reply: "
                "'Sorry, I'm a music chatbot. I can only help you with music recommendations and related discussions.'"
            )

            st.session_state.gemini_chat = model.start_chat(history=[
                {"role": "user", "parts": [system_prompt]},
                *[
                    {"role": msg["role"], "parts": [msg["content"]]}
                for msg in st.session_state["messages"]
                 ]
            ])

        else:
            system_prompt = (
                "You are a helpful AI chatbot specialized in music. "
                "You can recommend songs, discuss genres, suggest playlists based on mood, and talk about artists. "
                "If the user asks anything unrelated to music (like weather, sports, politics, or general chit-chat), reply: "
                "'Sorry, I'm a music chatbot. I can only help you with music recommendations and related discussions.'"
            )

            st.session_state.gemini_chat.history = [
                {"role": "user", "parts": [system_prompt]},
                *[
                    {"role": msg["role"], "parts": [msg["content"]]}
                    for msg in st.session_state["messages"]
                ]
            ]
            should_recommend = any(word in user_message.lower() for word in ["sad", "happy", "energetic", "calm", "relax", "depressed", "excited", "bored"])
            if should_recommend:
                sentiment = analyze_sentiment(user_message)
                song_names = recommend_music_list(sentiment)  # just names, not full text
                spotify_data = search_spotify_tracks(song_names)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"🎵 Based on your mood ({sentiment}), here are some Spotify tracks for you:"
                })

                # Show as table
                st.write("### Recommended Tracks")
                for track in spotify_data:
                    st.markdown(f"**{track['Song']}** by *{track['Artist']}* — [Listen on Spotify]({track['Link']})")


        # Send message with context-aware memory
        response = st.session_state.gemini_chat.send_message(user_message)
        bot_reply = response.text if response else "Sorry, I couldn't generate a response."
        placeholder.markdown(bot_reply)

    # Append assistant reply
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    save_chat_history(st.session_state["chat_file"], st.session_state["messages"])

    
    # Modify chat input handling to ask for preferences if it's the start of the session
    if len(st.session_state["messages"]) == 0:
        initial_prompt = "What type of music are you feeling like today? Are you in the mood for something upbeat, or more calm?"
        st.session_state["messages"].append({"role": "assistant", "content": initial_prompt})

    # Display the new prompt asking for user preferences
    st.subheader("Tell me what kind of music you're in the mood for 🎶")


    # 🔁 Automatically rename session file if not named
    if len(st.session_state["messages"]) >= 4 and not st.session_state.get("named"):
        session_name = generate_session_name(st.session_state["messages"][:6])
        safe_name = session_name.replace(" ", "_").replace("/", "_")
        new_file_path = os.path.join(user_folder, f"{safe_name}.json")

        # Avoid overwriting if exists
        counter = 1
        while os.path.exists(new_file_path):
            new_file_path = os.path.join(user_folder, f"{safe_name}_{counter}.json")
            counter += 1

        os.rename(st.session_state["chat_file"], new_file_path)
        st.session_state["chat_file"] = new_file_path
        st.session_state["named"] = False


st.write("---")
st.markdown("<p style='text-align: center;'>🎧 Happy Chatting! 🎧</p>", unsafe_allow_html=True)

def recommend_music_list(sentiment):
    if sentiment == "happy":
        return ["Happy - Pharrell Williams", "Can't Stop the Feeling - Justin Timberlake", "Levitating - Dua Lipa"]
    elif sentiment == "energetic":
        return ["Titanium - David Guetta", "Blinding Lights - The Weeknd", "Lose Yourself - Eminem"]
    elif sentiment == "sad":
        return ["Someone Like You - Adele", "Let Her Go - Passenger", "Fix You - Coldplay"]
    else:
        return ["Shape of You - Ed Sheeran", "Perfect - Ed Sheeran"]
