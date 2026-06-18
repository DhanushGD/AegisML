# src/simulate_traffic.py
import requests
import time

# URL of your local running FastAPI server
API_URL = "http://127.0.0.1:8000/predict"

# Simulating 5 distinct text entries that look different from the baseline training distribution
traffic_data = [
    "yo this update is actually fire ngl",
    "yo this update is actually fire ngl",
    "stfu bro you are literally capping right now",
    "bruh no cap this is the best discord server ever",
    "wtffff that stream was insane, go check out his clips",
    "can someone tell me how to bypass the login error on this app?",
    "I will kick your ass for sure!",
    "Honestly, I think we should be nicer to people online.",
    "This game is complete trash, the devs are brainless idiots.",
    "Let's meet up in the lobby and carry the team.",
    "Get out of our country, you don't belong here."
    "bro why is everyone talking about this rn",
    "nah that game was lowkey broken from the start",
    "just saw the craziest clip on my feed",
    "lmao who thought this was a good idea",
    "this app keeps crashing every five minutes",
    "fr this is the funniest thing ive seen today",
    "anyone know why my account got locked?",
    "that new song is stuck in my head",
    "cant believe people still fall for that scam",
    "bro really thought nobody would notice",
    "wtf is happening in this comment section",
    "just dropped a new video go watch it",
    "nah im done this makes no sense",
    "why is the server lagging so much today",
    "that was the cleanest play ive ever seen",
    "someone explain this meme to me please",
    "im crying this is way too funny",
    "this feature should have been added years ago",
    "who keeps spamming the chat like that",
    "bruh my wifi died in the middle of the match",
    "that ending was actually wild",
    "anybody else getting this weird error message",
    "this community is lowkey toxic sometimes",
    "cant stop watching these clips",
    "bro got absolutely destroyed in that debate",
    "why does this bug still exist",
    "the comments are funnier than the post",
    "i swear this algorithm knows everything",
    "that tutorial saved me hours of work",
    "nah this has to be bait",
    "who else is online at this hour",
    "just got the update and everything looks different",
    "bro really said that with confidence",
    "this stream is pure chaos right now",
    "my notifications have been blowing up all day",
    "why is everyone suddenly using this app",
    "that was the luckiest win ever",
    "im convinced nobody reads the instructions",
    "this challenge is harder than it looks",
    "someone please fix the audio issue",
    "cant tell if this is real or satire",
    "bro unlocked a new level of weird",
    "that reaction was priceless",
    "the chat is moving way too fast",
    "who remembers when this platform was actually simple",
    "just finished the event and it was amazing",
    "nah there is no way that actually happened",
    "this trend is getting out of control",
    "bro thats an instant unfollow from me",
    "you have absolutely no idea what you're talking about",
    "this take is unbelievably bad",
    "stop spreading nonsense everywhere",
    "that was the dumbest thing ive read all day",
    "why are people still arguing about this",
    "nobody asked for your opinion",
    "this comment section is a disaster",
    "you clearly didnt even read the article",
    "im tired of seeing the same argument over and over",
    "that explanation makes zero sense",
    "please stop posting misinformation",
    "what a complete waste of time",
    "this is getting ridiculous now",
    "you missed the point entirely",
    "im not sure how anyone could believe that",
    "the amount of confidence and lack of knowledge is impressive",
    "this discussion went downhill fast",
    "thats not how any of this works",
    "can people fact check before posting",
    "this thread is painful to read",
    "i regret opening the comments",
    "some of these replies are wild",
    "that argument falls apart immediately",
    "you keep repeating the same thing",
    "this conversation is going nowhere",
    "thats a terrible recommendation",
    "i dont understand the hype around this",
    "everyone needs to calm down",
    "the toxicity here is unreal",
    "this platform rewards bad behavior",
    "that was unnecessarily rude",
    "you dont need to insult people to make a point",
    "the replies are getting out of hand",
    "why is everyone so aggressive today",
    "this debate stopped being productive hours ago",
    "people are attacking each other instead of discussing ideas",
    "thats an unfair characterization",
    "i strongly disagree with that conclusion",
    "you are ignoring all the evidence",
    "that claim is unsupported",
    "someone explain the logic behind this",
    "this post is clearly rage bait",
    "the comment was removed for violating the rules",
    "moderators are probably going to lock this thread",
    "that response was completely uncalled for",
    "im muting this conversation",
    "there is way too much hostility here",
    "lets focus on the actual topic",
    "the personal attacks arent helping anyone",
    "this exchange became toxic very quickly"
]
print(f"🚥 Simulating live production traffic with {len(traffic_data)} comments...")
print("ℹ️  Note: Live predictions are automatically recorded to SQLite by the FastAPI server logic.\n")

success_count = 0

for i, text in enumerate(traffic_data):
    payload = {"text": text}
    try:
        # Call your live running FastAPI microservice
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            success_count += 1
            # Print a progress update for every 5th comment processed
            if (i + 1) % 5 == 0 or (i + 1) == len(traffic_data):
                print(f"✅ Processed {i+1}/{len(traffic_data)}: Model classified text and logged successfully.")
        else:
            print(f"⚠️ Server returned error for line {i}: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to the FastAPI server.")
        print("💡 Make sure your server is active in another terminal by running: uvicorn src.app:app --reload")
        break

    # Add a slight delay to mimic human behavior and avoid network choking
    time.sleep(0.1)

print(f"\n📈 Simulation finished! Successfully routed {success_count}/{len(traffic_data)} transactions directly into the production ecosystem.")