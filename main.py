import json
import time
import os
from datetime import datetime, timezone
import requests
import feedparser
from groq import Groq
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Configuration
subreddits = ["startups", "Entrepreneur", "StartupIdeas", "SaaS"]
output_file = "subsignal_top_posts.json"

# API Keys - Set these as environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# No Reddit authentication needed - using public RSS feeds

# Initialize API clients
groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# User agent for RSS requests
USER_AGENT = "SubSignalBot/2.0 (Startup Idea Analyzer)"

def fetch_reddit_posts():
    """Fetch posts from subreddits using public RSS feeds."""
    all_data = {}
    now_ts = int(datetime.now(timezone.utc).timestamp())
    
    headers = {'User-Agent': USER_AGENT}
    
    for sub in subreddits:
        try:
            print(f"Fetching posts from r/{sub}...")
            
            # Use Reddit's public RSS feed (no authentication required)
            rss_url = f"https://www.reddit.com/r/{sub}/new/.rss?limit=50"
            
            response = requests.get(rss_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse RSS feed
            feed = feedparser.parse(response.content)
            
            recent_posts = []
            for entry in feed.entries:
                # Parse published time
                try:
                    from time import mktime
                    from email.utils import parsedate
                    
                    published_ts = mktime(entry.published_parsed) if hasattr(entry, 'published_parsed') else 0
                    
                    # Filter posts from last 24 hours
                    if now_ts - published_ts <= 24*3600:
                        # Extract text content (RSS gives HTML)
                        content = entry.get('content', [{}])[0].get('value', '') if 'content' in entry else entry.get('summary', '')
                        
                        # Clean HTML tags
                        import re
                        clean_content = re.sub('<[^<]+?>', '', content)
                        
                        recent_posts.append({
                            "title": entry.title,
                            "body": clean_content[:500] if clean_content else "",
                            "score": 0,  # RSS doesn't provide scores
                            "url": entry.link,
                            "num_comments": 0,  # RSS doesn't provide comment counts
                            "created_utc": published_ts
                        })
                except Exception as e:
                    print(f"  ⚠ Error parsing entry: {e}")
                    continue
            
            # Take most recent posts (since we can't sort by score from RSS)
            recent_posts = recent_posts[:5]  # Get more since we can't filter by score
            
            all_data[sub] = recent_posts
            print(f"  ✓ Found {len(recent_posts)} posts from last 24 hours")
            
            # Be respectful with rate limiting
            time.sleep(3)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from r/{sub}: {e}")
            all_data[sub] = []
        except Exception as e:
            print(f"Unexpected error for r/{sub}: {e}")
            all_data[sub] = []
    
    return all_data

def prepare_groq_prompt(posts, subreddit):
    """Prepare compact prompt for GROQ with essential data only."""
    prompt = f"Analyze these 5 startup/business ideas from r/{subreddit} and select the ONE most promising project:\n\n"
    
    for i, post in enumerate(posts, 1):
        prompt += f"{i}. Title: {post['title']}\n"
        if post['body']:
            prompt += f"   Description: {post['body'][:300]}\n"
        prompt += f"   Score: {post['score']} | Comments: {post['num_comments']}\n\n"
    
    prompt += """Select the BEST idea based on:
- Innovation & market potential
- Feasibility
- Problem-solving impact
- Community engagement (score/comments)

Respond in JSON format:
{
  "selected_number": <1-5>,
  "title": "<title>",
  "reasoning": "<brief explanation why this is the best>"
}"""
    
    return prompt

def ask_groq_select_idea(posts, subreddit):
    """Send posts to GROQ and get the selected idea."""
    try:
        print(f"Asking GROQ to select best idea from r/{subreddit}...")
        prompt = prepare_groq_prompt(posts, subreddit)
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert startup analyst. Analyze ideas and select the most promising one. You MUST respond with valid JSON only, no additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        response_text = chat_completion.choices[0].message.content
        print(f"Debug - GROQ response: {response_text[:200]}...")
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        # Parse JSON response
        response_json = json.loads(response_text.strip())
        selected_idx = response_json["selected_number"] - 1
        
        # Validate index
        if selected_idx < 0 or selected_idx >= len(posts):
            print(f"⚠ Invalid index {selected_idx}, using first post")
            selected_idx = 0
        
        selected_idea = {
            "subreddit": subreddit,
            "title": posts[selected_idx]["title"],
            "body": posts[selected_idx]["body"],
            "url": posts[selected_idx]["url"],
            "score": posts[selected_idx]["score"],
            "groq_reasoning": response_json.get("reasoning", "No reasoning provided")
        }
        
        print(f"✓ Selected: {selected_idea['title'][:60]}...")
        return selected_idea
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error for r/{subreddit}: {e}")
        print(f"Response was: {response_text if 'response_text' in locals() else 'No response'}")
        # Fallback: return highest scored post
        return {
            "subreddit": subreddit,
            "title": posts[0]["title"],
            "body": posts[0]["body"],
            "url": posts[0]["url"],
            "score": posts[0]["score"],
            "groq_reasoning": "Selected by score (JSON parsing failed)"
        }
    except Exception as e:
        print(f"Error with GROQ for r/{subreddit}: {e}")
        # Fallback: return highest scored post
        return {
            "subreddit": subreddit,
            "title": posts[0]["title"],
            "body": posts[0]["body"],
            "url": posts[0]["url"],
            "score": posts[0]["score"],
            "groq_reasoning": "Selected by score (GROQ error)"
        }

def ask_gemini_rank_ideas(selected_ideas):
    """Send 4 selected ideas to Gemini for ranking and analysis."""
    try:
        print("\nAsking Gemini to rank and analyze the 4 selected ideas...")
        
        prompt = "You are an expert startup investor and analyst. Rank these 4 startup ideas and provide comprehensive analysis:\n\n"
        
        for i, idea in enumerate(selected_ideas, 1):
            prompt += f"{i}. [{idea['subreddit']}] {idea['title']}\n"
            if idea['body']:
                prompt += f"   Description: {idea['body'][:400]}\n"
            prompt += f"   Reddit Score: {idea['score']}\n"
            prompt += f"   GROQ's Selection Reasoning: {idea['groq_reasoning']}\n\n"
        
        prompt += """Provide detailed analysis in JSON format:
{
  "rankings": [
    {
      "rank": 1,
      "title": "<title>",
      "subreddit": "<subreddit>",
      "validation_score": <1-10>,
      "market_potential": "<analysis>",
      "feasibility": "<analysis>",
      "competitive_advantage": "<analysis>",
      "future_outlook": "<analysis>",
      "key_risks": "<risks>",
      "recommendation": "<invest/pass/watch>"
    }
  ],
  "overall_analysis": "<summary of all ideas and market trends>"
}"""
        
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type='application/json'
            )
        )
        
        # Extract JSON from response
        response_text = response.text
        
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        analysis = json.loads(response_text.strip())
        print("✓ Gemini analysis completed")
        return analysis
        
    except Exception as e:
        print(f"Error with Gemini: {e}")
        # Fallback response
        return {
            "rankings": [
                {
                    "rank": i+1,
                    "title": idea["title"],
                    "subreddit": idea["subreddit"],
                    "validation_score": 7,
                    "market_potential": "Analysis unavailable",
                    "recommendation": "watch"
                } for i, idea in enumerate(selected_ideas)
            ],
            "overall_analysis": "Gemini analysis failed. Manual review recommended."
        }

def send_to_discord(analysis, selected_ideas):
    """Send final analysis to Discord webhook."""
    
    # Check if Discord webhook is configured
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == "None":
        print("\n⚠ Discord webhook URL not configured. Skipping Discord notification.")
        print("💡 Set DISCORD_WEBHOOK_URL environment variable to enable Discord integration.")
        return
    
    try:
        print("\nSending results to Discord...")
        
        # Helper function to truncate text safely
        def truncate(text, max_len=1024):
            if not text or text == "N/A":
                return "Not available"
            text = str(text)
            return text[:max_len-3] + "..." if len(text) > max_len else text
        
        # Create rich embed message
        embeds = []
        
        # Main analysis embed
        overall_analysis = analysis.get("overall_analysis", "AI-powered analysis of top Reddit startup ideas")
        main_embed = {
            "title": "🚀 SubSignal: Top Startup Ideas Analysis",
            "description": truncate(overall_analysis, 4096),
            "color": 0x5865F2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Powered by GROQ + Gemini AI"}
        }
        embeds.append(main_embed)
        
        # Add each ranked idea as separate embed
        for ranking in analysis["rankings"][:4]:
            idea_data = next((idea for idea in selected_ideas if idea["title"] == ranking["title"]), None)
            
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣"][ranking["rank"]-1]
            
            # Build fields with proper validation
            fields = [
                {"name": "📍 Subreddit", "value": f"r/{ranking.get('subreddit', 'unknown')}", "inline": True},
                {"name": "⭐ Validation", "value": f"{ranking.get('validation_score', 'N/A')}/10", "inline": True},
                {"name": "📊 Status", "value": ranking.get("recommendation", "watch").upper(), "inline": True}
            ]
            
            # Add detailed analysis fields (truncated to Discord limits)
            if ranking.get("market_potential"):
                fields.append({
                    "name": "💼 Market Potential",
                    "value": truncate(ranking["market_potential"], 1024),
                    "inline": False
                })
            
            if ranking.get("feasibility"):
                fields.append({
                    "name": "🔧 Feasibility",
                    "value": truncate(ranking["feasibility"], 1024),
                    "inline": False
                })
            
            if ranking.get("future_outlook"):
                fields.append({
                    "name": "🔮 Future Outlook",
                    "value": truncate(ranking["future_outlook"], 1024),
                    "inline": False
                })
            
            if ranking.get("key_risks"):
                fields.append({
                    "name": "⚠️ Key Risks",
                    "value": truncate(ranking["key_risks"], 1024),
                    "inline": False
                })
            
            idea_embed = {
                "title": truncate(f"{rank_emoji} Rank {ranking['rank']}: {ranking['title']}", 256),
                "color": [0xFFD700, 0xC0C0C0, 0xCD7F32, 0x808080][ranking["rank"]-1],
                "fields": fields
            }
            
            if idea_data and idea_data.get("url"):
                idea_embed["url"] = idea_data["url"]
                fields.insert(3, {
                    "name": "👍 Reddit Score",
                    "value": str(idea_data["score"]),
                    "inline": True
                })
            
            embeds.append(idea_embed)
        
        # Send embeds in batches to avoid size limits
        # First message: Main analysis
        payload_main = {
            "username": "SubSignal Bot",
            "embeds": [embeds[0]]  # Main analysis embed
        }
        
        try:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload_main)
            response.raise_for_status()
            print("✓ Main analysis sent")
        except requests.exceptions.HTTPError as e:
            print(f"⚠ Failed to send main analysis: {e.response.text[:200]}")
        
        # Send each idea as a separate message to avoid size limits
        for i, idea_embed in enumerate(embeds[1:], 1):
            payload_idea = {
                "username": "SubSignal Bot",
                "embeds": [idea_embed]
            }
            
            try:
                response = requests.post(DISCORD_WEBHOOK_URL, json=payload_idea)
                response.raise_for_status()
                print(f"✓ Rank {i} idea sent")
                time.sleep(0.5)  # Small delay to avoid rate limits
            except requests.exceptions.HTTPError as e:
                print(f"⚠ Failed to send rank {i}: {e.response.text[:200]}")
                continue
        
        print("✓ All results sent to Discord successfully!")
        
    except requests.exceptions.HTTPError as e:
        print(f"Discord HTTP Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text[:500]}")
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def main():
    """Main execution flow."""
    print("=" * 60)
    print("SubSignal: AI-Powered Startup Idea Analyzer")
    print("=" * 60)
    
    # Check API keys configuration
    missing_keys = []
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if not GEMINI_API_KEY:
        missing_keys.append("GEMINI_API_KEY")
    
    if missing_keys:
        print(f"\n❌ Error: Missing required API keys: {', '.join(missing_keys)}")
        print("\nPlease set the following environment variables:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nWindows (PowerShell):")
        print(f'  $env:GROQ_API_KEY="your_key_here"')
        print(f'  $env:GEMINI_API_KEY="your_key_here"')
        return
    
    # No Reddit authentication needed - using public RSS feeds
    print("ℹ️  Using Reddit's public RSS feeds (no authentication required)")
    
    if not DISCORD_WEBHOOK_URL:
        print("\n⚠ Warning: DISCORD_WEBHOOK_URL not set. Results won't be sent to Discord.")
        print("         Analysis will still be saved to JSON files.\n")
    
    # Step 1: Fetch Reddit posts
    print("\n[Step 1] Fetching Reddit posts from 4 subreddits...")
    reddit_data = fetch_reddit_posts()
    
    # Save raw data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reddit_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Raw data saved to {output_file}")
    
    # Step 2: Ask GROQ to select best idea from each subreddit (with 30s delay)
    print("\n[Step 2] Processing each subreddit with GROQ (30s delay between each)...")
    selected_ideas = []
    
    for i, (subreddit, posts) in enumerate(reddit_data.items()):
        if not posts:
            print(f"⚠ No posts found for r/{subreddit}, skipping...")
            continue
            
        selected_idea = ask_groq_select_idea(posts, subreddit)
        selected_ideas.append(selected_idea)
        
        # 30 second delay between groups (except after last one)
        if i < len(reddit_data) - 1:
            print(f"Waiting 30 seconds before next subreddit...")
            time.sleep(30)
    
    if not selected_ideas:
        print("❌ No ideas were selected. Exiting.")
        return
    
    # Save selected ideas
    with open("selected_ideas.json", "w", encoding="utf-8") as f:
        json.dump(selected_ideas, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {len(selected_ideas)} ideas selected and saved to selected_ideas.json")
    
    # Step 3: Ask Gemini to rank and analyze the selected ideas
    print("\n[Step 3] Sending to Gemini for ranking and analysis...")
    analysis = ask_gemini_rank_ideas(selected_ideas)
    
    # Save analysis
    with open("gemini_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print("✓ Analysis saved to gemini_analysis.json")
    
    # Step 4: Send to Discord
    print("\n[Step 4] Sending final results to Discord...")
    send_to_discord(analysis, selected_ideas)
    
    print("\n" + "=" * 60)
    print("✅ Process completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
