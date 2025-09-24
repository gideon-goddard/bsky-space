import os
from dotenv import load_dotenv
from atproto import Client
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Load environment variables
load_dotenv()
BLUESKY_HANDLE = os.getenv('BLUESKY_HANDLE')
BLUESKY_PASSWORD = os.getenv('BLUESKY_PASSWORD')

# Initialize Bluesky client
client = Client()
client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)

# Initialize sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_my_posts():
    # Fetch posts from your Bluesky feed
    feed = client.get_author_feed(BLUESKY_HANDLE)
    posts = []
    for item in feed.feed:
        try:
            text = item.post.record.text
            if text:
                posts.append(text)
        except AttributeError:
            continue
    return posts

def vectorize_posts(posts):
    # Vectorize post texts
    return model.encode(posts)

def process_firehose_post(post):
    # Vectorize a single post from firehose
    return model.encode([post])[0]

app = FastAPI()

@app.get("/vectors")
def get_vectors():
    posts = fetch_my_posts()
    vectors = vectorize_posts(posts)
    return JSONResponse({
        "vectors": vectors.tolist(),
        "posts": posts
    })

@app.get("/")
def index():
    # Simple Plotly 3D scatter plot HTML
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h2>3D Visualization of Your Bluesky Posts</h2>
        <div id="plot" style="width:100vw;height:90vh;"></div>
        <script>
        fetch('/vectors').then(r => r.json()).then(data => {
            let vectors = data.vectors;
            let posts = data.posts;
            // Use first 3 dimensions for 3D plot
            let x = vectors.map(v => v[0]);
            let y = vectors.map(v => v[1]);
            let z = vectors.map(v => v[2]);
            let text = posts;
            let trace = {
                x: x,
                y: y,
                z: z,
                mode: 'markers',
                type: 'scatter3d',
                text: text,
                marker: { size: 5 }
            };
            Plotly.newPlot('plot', [trace], {margin: {l:0,r:0,b:0,t:0}});
        });
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(html)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        uvicorn.run("vectorize_posts:app", host="0.0.0.0", port=8000, reload=True)
    else:
        posts = fetch_my_posts()
        if not posts:
            print("No posts found. Debugging feed response...")
            feed = client.get_author_feed(BLUESKY_HANDLE)
            print("Raw feed:", feed)
        else:
            vectors = vectorize_posts(posts)
            print(f"Fetched {len(posts)} posts. First vector: {vectors[0]}")
