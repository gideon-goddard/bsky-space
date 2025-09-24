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

def fetch_discover_posts(limit=50):
    # Fetch posts from the discover feed
    feed = client.get_timeline(limit=limit)
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
    my_posts = fetch_my_posts()
    my_vectors = vectorize_posts(my_posts)
    discover_posts = fetch_discover_posts(limit=len(my_posts))
    discover_vectors = vectorize_posts(discover_posts)
    # Get URIs for each post
    my_uris = []
    feed = client.get_author_feed(BLUESKY_HANDLE)
    for item in feed.feed:
        try:
            my_uris.append(item.post.uri)
        except AttributeError:
            my_uris.append("")
    discover_uris = []
    discover_feed = client.get_timeline(limit=len(my_posts))
    for item in discover_feed.feed:
        try:
            discover_uris.append(item.post.uri)
        except AttributeError:
            discover_uris.append("")
    return JSONResponse({
        "my_vectors": my_vectors.tolist(),
        "my_posts": my_posts,
        "my_uris": my_uris,
        "discover_vectors": discover_vectors.tolist(),
        "discover_posts": discover_posts,
        "discover_uris": discover_uris
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
        <h2>3D Visualization of Your Bluesky Posts vs Discover Feed</h2>
        <div id="plot" style="width:100vw;height:90vh;"></div>
        <script>
        function openPost(uri) {
            if (!uri) return;
            // Bluesky web link format
            let url = 'https://bsky.app/profile/' + uri.split('/')[2] + '/post/' + uri.split('/')[4];
            window.open(url, '_blank');
        }
        fetch('/vectors').then(r => r.json()).then(data => {
            let my_vectors = data.my_vectors;
            let my_posts = data.my_posts;
            let my_uris = data.my_uris;
            let discover_vectors = data.discover_vectors;
            let discover_posts = data.discover_posts;
            let discover_uris = data.discover_uris;
            // Use first 3 dimensions for 3D plot
            let x1 = my_vectors.map(v => v[0]);
            let y1 = my_vectors.map(v => v[1]);
            let z1 = my_vectors.map(v => v[2]);
            let x2 = discover_vectors.map(v => v[0]);
            let y2 = discover_vectors.map(v => v[1]);
            let z2 = discover_vectors.map(v => v[2]);
            let trace1 = {
                x: x1,
                y: y1,
                z: z1,
                mode: 'markers',
                type: 'scatter3d',
                text: my_posts.map((t, i) => `<b>My Post</b><br>${t}`),
                marker: { size: 5, color: 'blue' },
                name: 'My Posts',
                customdata: my_uris,
                hovertemplate: '%{text}<extra></extra>'
            };
            let trace2 = {
                x: x2,
                y: y2,
                z: z2,
                mode: 'markers',
                type: 'scatter3d',
                text: discover_posts.map((t, i) => `<b>Discover</b><br>${t}`),
                marker: { size: 5, color: 'red' },
                name: 'Discover Feed',
                customdata: discover_uris,
                hovertemplate: '%{text}<extra></extra>'
            };
            let layout = {margin: {l:0,r:0,b:0,t:0}};
            let config = {responsive: true};
            Plotly.newPlot('plot', [trace1, trace2], layout, config);
            var plotDiv = document.getElementById('plot');
            plotDiv.on('plotly_click', function(data){
                var point = data.points[0];
                var uri = point.customdata;
                openPost(uri);
            });
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
