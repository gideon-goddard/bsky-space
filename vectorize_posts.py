import os
from dotenv import load_dotenv
from atproto import Client
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from tqdm import tqdm
import numpy as np
from functools import lru_cache
import math

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
    print("Fetching all your posts...")
    posts = []
    cursor = None
    page = 1
    while True:
        print(f"Fetching page {page} of your posts...")
        feed = client.get_author_feed(BLUESKY_HANDLE, cursor=cursor)
        page_posts = []
        for item in tqdm(feed.feed, desc=f"Processing page {page} posts", leave=False):
            try:
                text = item.post.record.text
                if text:
                    posts.append(text)
                    page_posts.append(text)
            except AttributeError:
                continue
        print(f"Fetched {len(page_posts)} posts from page {page}.")
        cursor = getattr(feed, 'cursor', None)
        if not cursor:
            break
        page += 1
    print(f"Total posts fetched: {len(posts)}")
    return posts

def fetch_discover_posts(limit=50):
    print(f"Fetching up to {limit} posts from the discover feed...")
    posts = []
    cursor = None
    page = 1
    while len(posts) < limit:
        batch_limit = min(100, limit - len(posts))
        print(f"Fetching page {page} of discover feed (batch size: {batch_limit})...")
        feed = client.get_timeline(limit=batch_limit, cursor=cursor)
        page_posts = []
        for item in tqdm(feed.feed, desc=f"Processing discover page {page}", leave=False):
            try:
                text = item.post.record.text
                if text:
                    posts.append(text)
                    page_posts.append(text)
            except AttributeError:
                continue
        print(f"Fetched {len(page_posts)} discover posts from page {page}.")
        cursor = getattr(feed, 'cursor', None)
        if not cursor or not feed.feed:
            break
        page += 1
    print(f"Total discover posts fetched: {len(posts)}")
    return posts

def fetch_mutuals_posts(limit=50):
    print(f"Fetching up to {limit} posts from the mutuals feed...")
    posts = []
    uris = []
    cursor = None
    page = 1
    # Use the correct at-uri for the official Bluesky mutuals feed generator
    mutuals_feed_uri = 'at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/mutuals'
    while len(posts) < limit:
        batch_limit = min(100, limit - len(posts))
        print(f"Fetching page {page} of mutuals feed (batch size: {batch_limit})...")
        params = {'feed': mutuals_feed_uri, 'limit': batch_limit}
        if cursor:
            params['cursor'] = cursor
        feed = client.app.bsky.feed.get_feed(params)
        page_posts = []
        for item in tqdm(feed.feed, desc=f"Processing mutuals page {page}", leave=False):
            try:
                text = item.post.record.text
                if text:
                    posts.append(text)
                    page_posts.append(text)
                    uris.append(item.post.uri)
            except AttributeError:
                continue
        print(f"Fetched {len(page_posts)} mutuals posts from page {page}.")
        cursor = getattr(feed, 'cursor', None)
        if not cursor or not feed.feed:
            break
        page += 1
    print(f"Total mutuals posts fetched: {len(posts)}")
    return posts, uris

def vectorize_posts(posts):
    print(f"Vectorizing {len(posts)} posts...")
    vectors = model.encode(posts, show_progress_bar=True)
    print("Vectorization complete.")
    return vectors

def process_firehose_post(post):
    # Vectorize a single post from firehose
    return model.encode([post])[0]

@lru_cache(maxsize=1)
def get_my_data():
    print("Fetching and vectorizing all your posts (cached)...")
    posts = fetch_my_posts()[:50]  # Limit to 50 posts for development
    vectors = vectorize_posts(posts)
    print("Fetching URIs for your posts...")
    uris = []
    cursor = None
    total_uris = 0
    with tqdm(total=len(posts), desc="Fetching URIs for your posts") as pbar:
        while total_uris < len(posts):
            batch_limit = min(100, len(posts) - total_uris)
            feed = client.get_author_feed(BLUESKY_HANDLE, cursor=cursor)
            for item in tqdm(feed.feed, desc=f"Processing URI batch", leave=False):
                try:
                    uris.append(item.post.uri)
                except AttributeError:
                    uris.append("")
            cursor = getattr(feed, 'cursor', None)
            total_uris += len(feed.feed)
            pbar.update(len(feed.feed))
            if not cursor or not feed.feed:
                break
    # Fetch discover posts and vectors for cross-set distance calculation
    discover_posts = fetch_discover_posts(limit=len(posts))
    discover_vectors = vectorize_posts(discover_posts)
    print("Fetching URIs for discover posts...")
    discover_uris = []
    cursor = None
    total_uris = 0
    with tqdm(total=len(discover_posts), desc="Fetching URIs for discover posts") as pbar:
        while total_uris < len(discover_posts):
            batch_limit = min(100, len(discover_posts) - total_uris)
            feed = client.get_timeline(limit=batch_limit, cursor=cursor)
            for item in tqdm(feed.feed, desc=f"Processing discover URI batch", leave=False):
                try:
                    discover_uris.append(item.post.uri)
                except AttributeError:
                    discover_uris.append("")
            cursor = getattr(feed, 'cursor', None)
            total_uris += len(feed.feed)
            pbar.update(len(feed.feed))
            if not cursor or not feed.feed:
                break
    # Fetch mutuals posts and vectors
    mutuals_posts, mutuals_uris = fetch_mutuals_posts(limit=len(posts))
    mutuals_vectors = vectorize_posts(mutuals_posts)
    # Calculate cross-set distances
    print("Calculating cross-set closest/farthest pairs...")
    arr_my = np.array(vectors)
    arr_discover = np.array(discover_vectors)
    arr_mutuals = np.array(mutuals_vectors)
    n_my = len(arr_my)
    n_discover = len(arr_discover)
    n_mutuals = len(arr_mutuals)
    min_dist = float('inf')
    max_dist = float('-inf')
    min_pair = None
    max_pair = None
    min_dist_mutuals = float('inf')
    max_dist_mutuals = float('-inf')
    min_pair_mutuals = None
    max_pair_mutuals = None
    for i in tqdm(range(n_my), desc="Cross-set pairwise distance calculation"):
        for j in range(n_discover):
            dist = np.linalg.norm(arr_my[i] - arr_discover[j])
            if dist < min_dist:
                min_dist = dist
                min_pair = (i, j)
            if dist > max_dist:
                max_dist = dist
                max_pair = (i, j)
    for i in tqdm(range(len(vectors)), desc="Pairwise distance to mutuals"):
        for j in range(n_mutuals):
            dist = np.linalg.norm(np.array(vectors[i]) - arr_mutuals[j])
            if dist < min_dist_mutuals:
                min_dist_mutuals = dist
                min_pair_mutuals = (i, j)
            if dist > max_dist_mutuals:
                max_dist_mutuals = dist
                max_pair_mutuals = (i, j)
    closest = {
        "indices": (int(min_pair[0]), int(min_pair[1])),
        "distance": float(min_dist),
        "my_post": posts[min_pair[0]],
        "my_uri": uris[min_pair[0]],
        "discover_post": discover_posts[min_pair[1]],
        "discover_uri": discover_uris[min_pair[1]]
    }
    farthest = {
        "indices": (int(max_pair[0]), int(max_pair[1])),
        "distance": float(max_dist),
        "my_post": posts[max_pair[0]],
        "my_uri": uris[max_pair[0]],
        "discover_post": discover_posts[max_pair[1]],
        "discover_uri": discover_uris[max_pair[1]]
    }
    closest_mutuals = {
        "indices": (int(min_pair_mutuals[0]), int(min_pair_mutuals[1])),
        "distance": float(min_dist_mutuals),
        "my_post": posts[min_pair_mutuals[0]],
        "my_uri": uris[min_pair_mutuals[0]],
        "mutuals_post": mutuals_posts[min_pair_mutuals[1]],
        "mutuals_uri": mutuals_uris[min_pair_mutuals[1]]
    }
    farthest_mutuals = {
        "indices": (int(max_pair_mutuals[0]), int(max_pair_mutuals[1])),
        "distance": float(max_dist_mutuals),
        "my_post": posts[max_pair_mutuals[0]],
        "my_uri": uris[max_pair_mutuals[0]],
        "mutuals_post": mutuals_posts[max_pair_mutuals[1]],
        "mutuals_uri": mutuals_uris[max_pair_mutuals[1]]
    }
    print("Cross-set pair calculations complete.")
    return {
        "posts": posts,
        "vectors": vectors,
        "uris": uris,
        "discover_posts": discover_posts,
        "discover_vectors": discover_vectors,
        "discover_uris": discover_uris,
        "mutuals_posts": mutuals_posts,
        "mutuals_vectors": mutuals_vectors,
        "mutuals_uris": mutuals_uris,
        "closest": closest,
        "farthest": farthest,
        "closest_mutuals": closest_mutuals,
        "farthest_mutuals": farthest_mutuals
    }

def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        if math.isinf(v) or math.isnan(v):
            return None
        return v
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    else:
        return obj

app = FastAPI()

@app.get("/vectors")
def get_vectors():
    data = get_my_data()
    discover_posts = fetch_discover_posts(limit=len(data["posts"]))
    discover_vectors = vectorize_posts(discover_posts)
    discover_uris = []
    cursor = None
    total_uris = 0
    with tqdm(total=len(discover_posts), desc="Fetching URIs for discover posts") as pbar:
        while total_uris < len(discover_posts):
            batch_limit = min(100, len(discover_posts) - total_uris)
            feed = client.get_timeline(limit=batch_limit, cursor=cursor)
            for item in tqdm(feed.feed, desc=f"Processing discover URI batch", leave=False):
                try:
                    discover_uris.append(item.post.uri)
                except AttributeError:
                    discover_uris.append("")
            cursor = getattr(feed, 'cursor', None)
            total_uris += len(feed.feed)
            pbar.update(len(feed.feed))
            if not cursor or not feed.feed:
                break
    # Fetch mutuals posts for the endpoint
    mutuals_posts, mutuals_uris = fetch_mutuals_posts(limit=len(data["posts"]))
    mutuals_vectors = vectorize_posts(mutuals_posts)
    result = {
        "my_vectors": data["vectors"],
        "my_posts": data["posts"],
        "my_uris": data["uris"],
        "discover_vectors": discover_vectors,
        "discover_posts": discover_posts,
        "discover_uris": discover_uris,
        "mutuals_vectors": mutuals_vectors,
        "mutuals_posts": mutuals_posts,
        "mutuals_uris": mutuals_uris,
        "closest": data["closest"],
        "farthest": data["farthest"],
        "closest_mutuals": data["closest_mutuals"],
        "farthest_mutuals": data["farthest_mutuals"]
    }
    return JSONResponse(sanitize_json(result))

@app.get("/stats")
def get_stats():
    data = get_my_data()
    return JSONResponse({
        "closest": data["closest"],
        "farthest": data["farthest"],
        "closest_mutuals": data["closest_mutuals"],
        "farthest_mutuals": data["farthest_mutuals"]
    })

@app.get("/highlighted")
def highlighted():
    return JSONResponse({"highlighted": None})

@app.get("/")
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
        .pair-list { margin: 2em 0; }
        .pair-list li { margin-bottom: 1em; cursor: pointer; }
        .pair-list li.selected { background: #e0e0ff; }
        .pair-btn { margin-left: 1em; }
        </style>
    </head>
    <body>
        <h2>3D Visualization of Your Bluesky Posts vs Discover Feed</h2>
        <div id="plot" style="width:100vw;height:70vh;"></div>
        <div id="pairs" style="width:100vw;"></div>
        <script>
        let plotData, plotDiv;
        function uriToUrl(uri) {
            // Example: at://did:plc:xxxx/app.bsky.feed.post/yyyy
            let parts = uri.split('/');
            if (parts.length < 5) return '';
            let did = parts[2]; // includes 'did:plc:xxxx'
            let rkey = parts[4];
            return `https://bsky.app/profile/${did}/post/${rkey}`;
        }
        function openPost(uri) {
            let url = uriToUrl(uri);
            if (url) window.open(url, '_blank');
        }
        let highlightLine = null;
        function highlightPair(idxA, idxB) {
            // Highlight points in Plotly
            let colors = plotData[0].x.map(_ => 'blue');
            let colors2 = plotData[1].x.map(_ => 'red');
            // Reset all nodes
            for (let i = 0; i < colors.length; i++) colors[i] = 'blue';
            for (let i = 0; i < colors2.length; i++) colors2[i] = 'red';
            // Highlight selected nodes
            let isClosest = document.getElementById('Closest').classList.contains('selected');
            let nodeColor = isClosest ? 'lime' : 'orange';
            colors[idxA] = nodeColor;
            colors2[idxB] = nodeColor;
            Plotly.restyle('plot', { marker: { color: colors } }, [0]);
            Plotly.restyle('plot', { marker: { color: colors2 } }, [1]);
            // Remove previous highlight line
            if (highlightLine !== null) {
                Plotly.deleteTraces('plot', highlightLine);
                highlightLine = null;
            }
            // Draw line between selected pair
            if (typeof idxA === 'number' && typeof idxB === 'number') {
                let xA = plotData[0].x[idxA], yA = plotData[0].y[idxA], zA = plotData[0].z[idxA];
                let xB = plotData[1].x[idxB], yB = plotData[1].y[idxB], zB = plotData[1].z[idxB];
                let lineColor = nodeColor;
                let lineTrace = {
                    x: [xA, xB],
                    y: [yA, yB],
                    z: [zA, zB],
                    mode: 'lines',
                    type: 'scatter3d',
                    line: { color: lineColor, width: 6 },
                    showlegend: false
                };
                Plotly.addTraces('plot', lineTrace);
                highlightLine = plotData.length; // index of the new trace
            }
        }
        function highlightPair_mutuals(idxA, idxB) {
            // Highlight points in Plotly for mutuals
            let colors = plotData[0].x.map(_ => 'blue');
            let colors2 = plotData[2].x.map(_ => 'purple');
            for (let i = 0; i < colors.length; i++) colors[i] = 'blue';
            for (let i = 0; i < colors2.length; i++) colors2[i] = 'purple';
            let isClosest = document.getElementById('Closestmutuals').classList.contains('selected');
            let nodeColor = isClosest ? 'lime' : 'orange';
            colors[idxA] = nodeColor;
            colors2[idxB] = nodeColor;
            Plotly.restyle('plot', { marker: { color: colors } }, [0]);
            Plotly.restyle('plot', { marker: { color: colors2 } }, [2]);
            // Remove previous highlight line
            if (highlightLine !== null) {
                Plotly.deleteTraces('plot', highlightLine);
                highlightLine = null;
            }
            // Draw line between selected pair
            if (typeof idxA === 'number' && typeof idxB === 'number') {
                let xA = plotData[0].x[idxA], yA = plotData[0].y[idxA], zA = plotData[0].z[idxA];
                let xB = plotData[2].x[idxB], yB = plotData[2].y[idxB], zB = plotData[2].z[idxB];
                let lineColor = nodeColor;
                let lineTrace = {
                    x: [xA, xB],
                    y: [yA, yB],
                    z: [zA, zB],
                    mode: 'lines',
                    type: 'scatter3d',
                    line: { color: lineColor, width: 6 },
                    showlegend: false
                };
                Plotly.addTraces('plot', lineTrace);
                highlightLine = plotData.length;
            }
        }
        fetch('/vectors').then(r => r.json()).then(data => {
            plotData = [];
            let my_vectors = data.my_vectors;
            let my_posts = data.my_posts;
            let my_uris = data.my_uris;
            let discover_vectors = data.discover_vectors;
            let discover_posts = data.discover_posts;
            let discover_uris = data.discover_uris;
            let mutuals_vectors = data.mutuals_vectors;
            let mutuals_posts = data.mutuals_posts;
            let mutuals_uris = data.mutuals_uris;
            let x1 = my_vectors.map(v => v[0]);
            let y1 = my_vectors.map(v => v[1]);
            let z1 = my_vectors.map(v => v[2]);
            let x2 = discover_vectors.map(v => v[0]);
            let y2 = discover_vectors.map(v => v[1]);
            let z2 = discover_vectors.map(v => v[2]);
            let x3 = mutuals_vectors.map(v => v[0]);
            let y3 = mutuals_vectors.map(v => v[1]);
            let z3 = mutuals_vectors.map(v => v[2]);
            plotData = [
                {
                    x: x1,
                    y: y1,
                    z: z1,
                    mode: 'markers',
                    type: 'scatter3d',
                    text: my_posts.map((t, i) => `<b>My Post</b><br>${t}`),
                    marker: { size: 5, color: Array(x1.length).fill('blue') },
                    name: 'My Posts',
                    customdata: my_uris,
                    hovertemplate: '%{text}<extra></extra>'
                },
                {
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
                },
                {
                    x: x3,
                    y: y3,
                    z: z3,
                    mode: 'markers',
                    type: 'scatter3d',
                    text: mutuals_posts.map((t, i) => `<b>Mutuals</b><br>${t}`),
                    marker: { size: 5, color: Array(x3.length).fill('purple') },
                    name: 'Mutuals Feed',
                    customdata: mutuals_uris,
                    hovertemplate: '%{text}<extra></extra>'
                }
            ];
            let layout = {margin: {l:0,r:0,b:0,t:0}};
            let config = {responsive: true};
            Plotly.newPlot('plot', plotData, layout, config);
            plotDiv = document.getElementById('plot');
            plotDiv.on('plotly_click', function(data){
                var point = data.points[0];
                var uri = point.customdata;
                openPost(uri);
            });
            // Fetch closest/farthest pairs
            fetch('/stats').then(r => r.json()).then(stats => {
                let html = '<h3>Closest and Farthest Pairs</h3><ul class="pair-list">';
                function pairItem(pair, label, feedLabel) {
                    let idxA = pair.indices[0], idxB = pair.indices[1];
                    let my_post = pair.my_post, other_post = pair[feedLabel + '_post'];
                    let my_uri = pair.my_uri, other_uri = pair[feedLabel + '_uri'];
                    return `<li id="${label + feedLabel}" onclick="highlightPair_${feedLabel}(${idxA},${idxB});this.classList.add('selected');">
                        <b>${label} pair (${feedLabel})</b> (distance: ${pair.distance.toFixed(4)})<br>
                        <span><a href="#" onclick=\"event.stopPropagation();openPost(\"${my_uri}\")\">My Post: ${my_post}</a></span><br>
                        <span><a href="#" onclick=\"event.stopPropagation();openPost(\"${other_uri}\")\">${feedLabel.charAt(0).toUpperCase() + feedLabel.slice(1)}: ${other_post}</a></span>
                    </li>`;
                }
                html += pairItem(stats.closest, 'Closest', 'discover');
                html += pairItem(stats.farthest, 'Farthest', 'discover');
                html += pairItem(stats.closest_mutuals, 'Closest', 'mutuals');
                html += pairItem(stats.farthest_mutuals, 'Farthest', 'mutuals');
                html += '</ul>';
                document.getElementById('pairs').innerHTML = html;
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
