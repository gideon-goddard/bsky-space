import os
from dotenv import load_dotenv
from atproto import Client
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from tqdm import tqdm
import numpy as np
from functools import lru_cache
import math
from sklearn.decomposition import PCA

# Load environment variables
load_dotenv()
BLUESKY_HANDLE = os.getenv('BLUESKY_HANDLE')
BLUESKY_PASSWORD = os.getenv('BLUESKY_PASSWORD')

# Initialize Bluesky client
client = Client()
client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)

# Initialize sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_my_posts_with_uris():
    print("Fetching all your posts with URIs...")
    posts = []
    uris = []
    cursor = None
    page = 1
    while True:
        print(f"Fetching page {page} of your posts...")
        feed = client.get_author_feed(BLUESKY_HANDLE, cursor=cursor)
        for item in tqdm(feed.feed, desc=f"Processing page {page} posts", leave=False):
            try:
                # Skip reposts
                if hasattr(item, 'reason') and getattr(item.reason, '$type', None) == "app.bsky.feed.defs#reasonRepost":
                    continue
                text = item.post.record.text
                uri = item.post.uri
                if text:
                    posts.append(text)
                    uris.append(uri)
            except AttributeError:
                continue
        cursor = getattr(feed, 'cursor', None)
        if not cursor:
            break
        page += 1
    print(f"Total posts fetched: {len(posts)}")
    return posts, uris

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
                # Skip reposts
                if hasattr(item, 'reason') and getattr(item.reason, '$type', None) == "app.bsky.feed.defs#reasonRepost":
                    continue
                text = item.post.record.text
                author = getattr(item.post, 'author', None)
                did = getattr(author, 'did', None) if author else None
                # Filter out posts by the current user and reposts
                is_repost = getattr(item.post.record, 'reply', None) is not None or getattr(item.post.record, 'repost', None) is not None
                if text and did != client.me.did and not is_repost:
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

def vectorize_posts(posts, pca_model=None):
    print(f"Vectorizing {len(posts)} posts...")
    vectors = model.encode(posts, show_progress_bar=True)
    print("Vectorization complete.")
    if pca_model is not None:
        print("Reducing vectors to 3D with PCA...")
        vectors = pca_model.transform(vectors)
        print("PCA reduction complete.")
    return vectors

def process_firehose_post(post):
    # Vectorize a single post from firehose
    return model.encode([post])[0]

@lru_cache(maxsize=1)
def get_my_data():
    print("Fetching and vectorizing all your posts (cached)...")
    posts, uris = fetch_my_posts_with_uris()
    raw_vectors = model.encode(posts, show_progress_bar=True)
    mutuals_posts, mutuals_uris = fetch_mutuals_posts(limit=len(posts))
    raw_mutuals_vectors = model.encode(mutuals_posts, show_progress_bar=True)
    all_vectors = np.vstack([raw_vectors, raw_mutuals_vectors])
    print("Fitting PCA for 3D reduction...")
    pca = PCA(n_components=3)
    pca.fit(all_vectors)
    vectors = pca.transform(raw_vectors)
    mutuals_vectors = pca.transform(raw_mutuals_vectors)
    # Calculate cross-set closest/farthest pairs and distance matrix
    print("Calculating cross-set closest/farthest pairs and distance matrix...")
    arr_my = np.array(vectors)
    arr_mutuals = np.array(mutuals_vectors)
    n_my = len(arr_my)
    n_mutuals = len(arr_mutuals)
    # Pairwise distance matrix (my posts vs mutuals)
    distance_matrix = np.zeros((n_my, n_mutuals))
    min_dist_mutuals = float('inf')
    max_dist_mutuals = float('-inf')
    min_pair_mutuals = None
    max_pair_mutuals = None
    for i in range(n_my):
        for j in range(n_mutuals):
            dist = np.linalg.norm(arr_my[i] - arr_mutuals[j])
            distance_matrix[i, j] = dist
            if dist < min_dist_mutuals:
                min_dist_mutuals = dist
                min_pair_mutuals = (i, j)
            if dist > max_dist_mutuals:
                max_dist_mutuals = dist
                max_pair_mutuals = (i, j)
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
        "mutuals_posts": mutuals_posts,
        "mutuals_vectors": mutuals_vectors,
        "mutuals_uris": mutuals_uris,
        "closest_mutuals": closest_mutuals,
        "farthest_mutuals": farthest_mutuals,
        "distance_matrix": distance_matrix.tolist(),
        "pca": pca
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
    result = {
        "my_vectors": data["vectors"],
        "my_posts": data["posts"],
        "my_uris": data["uris"],
        "mutuals_vectors": data["mutuals_vectors"],
        "mutuals_posts": data["mutuals_posts"],
        "mutuals_uris": data["mutuals_uris"],
        "closest_mutuals": data["closest_mutuals"],
        "farthest_mutuals": data["farthest_mutuals"],
        "distance_matrix": data["distance_matrix"]
    }
    return JSONResponse(sanitize_json(result))

@app.get("/stats")
def get_stats():
    data = get_my_data()
    return JSONResponse({
        "closest_mutuals": data["closest_mutuals"],
        "farthest_mutuals": data["farthest_mutuals"]
    })

@app.get("/highlighted")
def highlighted():
    return JSONResponse({"highlighted": None})

@app.post("/vectorize_new_post")
async def vectorize_new_post(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    main_data = get_my_data()
    pca = main_data["pca"]
    new_vec = model.encode([text])[0]
    new_vec_3d = pca.transform([new_vec])[0].tolist()
    return JSONResponse({"vector": new_vec_3d})

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
        <h2>3D Visualization of Your Bluesky Posts vs Mutuals Feed</h2>
        <div id="plot" style="width:100vw;height:70vh;"></div>
        <div id="pairs" style="width:100vw;"></div>
        <form id="newPostForm" style="margin:2em 0;">
            <label for="newPostInput"><b>Test a new post:</b></label>
            <input type="text" id="newPostInput" style="width:40em;" placeholder="Type your post here...">
            <button type="submit">Add to graph</button>
        </form>
        <div id="newPostResults"></div>
        <script>
        let plotData, plotDiv, newPostTraceIdx = null;
        function uriToUrl(uri) {
            let parts = uri.split('/');
            if (parts.length < 5) return '';
            let did = parts[2];
            let rkey = parts[4];
            return `https://bsky.app/profile/${did}/post/${rkey}`;
        }
        function openPost(uri) {
            let url = uriToUrl(uri);
            if (url) window.open(url, '_blank');
        }
        let highlightLine = null;
        function clearHighlights() {
            let plotDiv = document.getElementById('plot');
            // Remove all extra traces (lines and new post) beyond the main scatter traces
            while (plotDiv.data.length > 2) {
                Plotly.deleteTraces('plot', plotDiv.data.length - 1);
            }
            highlightLine = null;
            newPostTraceIdx = null;
            if (plotData) {
                Plotly.restyle('plot', { marker: { color: Array(plotData[0].x.length).fill('blue'), size: 5 } }, [0]);
                Plotly.restyle('plot', { marker: { color: Array(plotData[1].x.length).fill('purple'), size: 5 } }, [1]);
            }
            document.querySelectorAll('.pair-list li.selected').forEach(li => li.classList.remove('selected'));
            document.getElementById('newPostResults').innerHTML = '';
        }
        function highlightPair(idxA, idxB) {
            clearHighlights();
            if (typeof idxA === 'number' && typeof idxB === 'number') {
                let xA = plotData[0].x[idxA], yA = plotData[0].y[idxA], zA = plotData[0].z[idxA];
                let xB = plotData[1].x[idxB], yB = plotData[1].y[idxB], zB = plotData[1].z[idxB];
                let lineColor = 'orange';
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
        let cachedVectors = null;
        fetch('/vectors').then(r => r.json()).then(data => {
            plotData = [];
            cachedVectors = data;
            let my_vectors = data.my_vectors;
            let my_posts = data.my_posts;
            let my_uris = data.my_uris;
            let mutuals_vectors = data.mutuals_vectors;
            let mutuals_posts = data.mutuals_posts;
            let mutuals_uris = data.mutuals_uris;
            let distance_matrix = data.distance_matrix;
            let x1 = my_vectors.map(v => v[0]);
            let y1 = my_vectors.map(v => v[1]);
            let z1 = my_vectors.map(v => v[2]);
            let x2 = mutuals_vectors.map(v => v[0]);
            let y2 = mutuals_vectors.map(v => v[1]);
            let z2 = mutuals_vectors.map(v => v[2]);
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
                    text: mutuals_posts.map((t, i) => `<b>Mutuals</b><br>${t}`),
                    marker: { size: 5, color: Array(x2.length).fill('purple') },
                    name: 'Mutuals Feed',
                    customdata: mutuals_uris,
                    hovertemplate: '%{text}<extra></extra>'
                }
            ];
            let layout = {
                margin: {l:0,r:0,b:0,t:0},
                scene: {
                    dragmode: 'orbit',
                    camera: {projection: {type: 'perspective'}},
                },
                autosize: true,
                modebar: {
                    orientation: 'v',
                    bgcolor: 'rgba(255,255,255,0.7)',
                    color: '#333',
                    activecolor: '#007bff',
                    position: 'bottom right',
                },
            };
            let config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['sendDataToCloud'],
                displaylogo: false,
            };
            Plotly.newPlot('plot', plotData, layout, config);
            plotDiv = document.getElementById('plot');
            plotDiv.on('plotly_click', function(data){
                var point = data.points[0];
                if (point && point.customdata) {
                    var uri = point.customdata;
                    openPost(uri);
                }
            });
            // Pair results UI
            fetch('/stats').then(r => r.json()).then(stats => {
                let html = '<h3>Closest and Farthest Pairs</h3><ul class="pair-list">';
                function pairItem(pair, label) {
                    let idxA = pair.indices[0], idxB = pair.indices[1];
                    let my_post = pair.my_post, other_post = pair.mutuals_post;
                    let my_uri = pair.my_uri, other_uri = pair.mutuals_uri;
                    let dist = distance_matrix[idxA][idxB];
                    return `<li id="${label}mutuals" onclick="clearHighlights();highlightPair(${idxA},${idxB});this.classList.add('selected');">
                        <b>${label} pair</b> (distance: ${dist.toFixed(4)})<br>
                        <span>
                            <a href='#' onclick="event.stopPropagation();highlightPair(${idxA},${idxB});return false;">My Post: ${my_post}</a>
                            <button class='pair-btn' onclick="event.stopPropagation();openPost('${my_uri}')">Open</button>
                        </span><br>
                        <span>
                            <a href='#' onclick="event.stopPropagation();highlightPair(${idxA},${idxB});return false;">Mutuals: ${other_post}</a>
                            <button class='pair-btn' onclick="event.stopPropagation();openPost('${other_uri}')">Open</button>
                        </span>
                    </li>`;
                }
                html += pairItem(stats.closest_mutuals, 'Closest');
                html += pairItem(stats.farthest_mutuals, 'Farthest');
                html += '</ul>';
                document.getElementById('pairs').innerHTML = html;
            });
            // New post form logic
            document.getElementById('newPostForm').onsubmit = async function(e) {
                e.preventDefault();
                clearHighlights();
                let postText = document.getElementById('newPostInput').value.trim();
                if (!postText) return;
                // Call backend to vectorize and project the new post
                let resp = await fetch('/vectorize_new_post', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: postText })
                });
                let result = await resp.json();
                let v = result.vector;
                // Add new post to plot in red
                let trace = {
                    x: [v[0]], y: [v[1]], z: [v[2]],
                    mode: 'markers', type: 'scatter3d',
                    marker: { size: 8, color: 'red' },
                    name: 'New Post',
                    text: [`<b>New Post</b><br>${postText}`],
                    hovertemplate: '%{text}<extra></extra>'
                };
                Plotly.addTraces('plot', trace);
                newPostTraceIdx = plotDiv.data.length - 1;
                // Calculate closest/farthest to my posts and mutuals
                let minDistMy = Infinity, maxDistMy = -Infinity, minIdxMy = -1, maxIdxMy = -1;
                for (let i = 0; i < my_vectors.length; ++i) {
                    let d = Math.sqrt(
                        Math.pow(v[0] - my_vectors[i][0], 2) +
                        Math.pow(v[1] - my_vectors[i][1], 2) +
                        Math.pow(v[2] - my_vectors[i][2], 2)
                    );
                    if (d < minDistMy) { minDistMy = d; minIdxMy = i; }
                    if (d > maxDistMy) { maxDistMy = d; maxIdxMy = i; }
                }
                let minDistMut = Infinity, maxDistMut = -Infinity, minIdxMut = -1, maxIdxMut = -1;
                for (let i = 0; i < mutuals_vectors.length; ++i) {
                    let d = Math.sqrt(
                        Math.pow(v[0] - mutuals_vectors[i][0], 2) +
                        Math.pow(v[1] - mutuals_vectors[i][1], 2) +
                        Math.pow(v[2] - mutuals_vectors[i][2], 2)
                    );
                    if (d < minDistMut) { minDistMut = d; minIdxMut = i; }
                    if (d > maxDistMut) { maxDistMut = d; maxIdxMut = i; }
                }
                // Draw lines for closest and then farthest pairs (order matters)
                let lineTraceClosestMy = {
                    x: [v[0], my_vectors[minIdxMy][0]],
                    y: [v[1], my_vectors[minIdxMy][1]],
                    z: [v[2], my_vectors[minIdxMy][2]],
                    mode: 'lines', type: 'scatter3d',
                    line: { color: 'green', width: 5 },
                    showlegend: false
                };
                let lineTraceFarthestMy = {
                    x: [v[0], my_vectors[maxIdxMy][0]],
                    y: [v[1], my_vectors[maxIdxMy][1]],
                    z: [v[2], my_vectors[maxIdxMy][2]],
                    mode: 'lines', type: 'scatter3d',
                    line: { color: 'red', width: 5 },
                    showlegend: false
                };
                let lineTraceClosestMut = {
                    x: [v[0], mutuals_vectors[minIdxMut][0]],
                    y: [v[1], mutuals_vectors[minIdxMut][1]],
                    z: [v[2], mutuals_vectors[minIdxMut][2]],
                    mode: 'lines', type: 'scatter3d',
                    line: { color: 'green', width: 5, dash: 'dot' },
                    showlegend: false
                };
                let lineTraceFarthestMut = {
                    x: [v[0], mutuals_vectors[maxIdxMut][0]],
                    y: [v[1], mutuals_vectors[maxIdxMut][1]],
                    z: [v[2], mutuals_vectors[maxIdxMut][2]],
                    mode: 'lines', type: 'scatter3d',
                    line: { color: 'red', width: 5, dash: 'dot' },
                    showlegend: false
                };
                Plotly.addTraces('plot', [lineTraceClosestMy, lineTraceFarthestMy, lineTraceClosestMut, lineTraceFarthestMut]);
                // Show results
                let html = `<h4>New Post Results</h4>`;
                html += `<b>Closest to My Posts:</b> (distance: ${minDistMy.toFixed(4)})<br>`;
                html += `<span><a href='#' onclick="openPost('${my_uris[minIdxMy]}');return false;">${my_posts[minIdxMy]}</a></span><br>`;
                html += `<b>Farthest from My Posts:</b> (distance: ${maxDistMy.toFixed(4)})<br>`;
                html += `<span><a href='#' onclick="openPost('${my_uris[maxIdxMy]}');return false;">${my_posts[maxIdxMy]}</a></span><br>`;
                html += `<b>Closest to Mutuals:</b> (distance: ${minDistMut.toFixed(4)})<br>`;
                html += `<span><a href='#' onclick="openPost('${mutuals_uris[minIdxMut]}');return false;">${mutuals_posts[minIdxMut]}</a></span><br>`;
                html += `<b>Farthest from Mutuals:</b> (distance: ${maxDistMut.toFixed(4)})<br>`;
                html += `<span><a href='#' onclick="openPost('${mutuals_uris[maxIdxMut]}');return false;">${mutuals_posts[maxIdxMut]}</a></span><br>`;
                document.getElementById('newPostResults').innerHTML = html;
            };

            // Utility: Fuzz a message by adding random special characters
            function fuzzMessage(msg, fuzzLevel) {
                const specials = '!@#$%^&*()_+-=[]{}|;:,.<>?/';
                let arr = msg.split('');
                for (let i = 0; i < fuzzLevel; ++i) {
                    let idx = Math.floor(Math.random() * (arr.length + 1));
                    let char = specials[Math.floor(Math.random() * specials.length)];
                    arr.splice(idx, 0, char);
                }
                return arr.join('');
            }

            // Fuzz and plot until far from all nodes
            async function fuzzUntilFar(msg, minDist = 10, maxTries = 50) {
                let my_vectors = cachedVectors.my_vectors;
                let mutuals_vectors = cachedVectors.mutuals_vectors;
                let bestMsg = msg, bestVec = null, bestDist = -Infinity;
                for (let fuzzLevel = 1; fuzzLevel <= maxTries; ++fuzzLevel) {
                    let fuzzed = fuzzMessage(msg, fuzzLevel);
                    let resp = await fetch('/vectorize_new_post', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: fuzzed })
                    });
                    let result = await resp.json();
                    let v = result.vector;
                    // Find closest distance to any node
                    let minDistAny = Infinity;
                    for (let i = 0; i < my_vectors.length; ++i) {
                        let d = Math.sqrt(
                            Math.pow(v[0] - my_vectors[i][0], 2) +
                            Math.pow(v[1] - my_vectors[i][1], 2) +
                            Math.pow(v[2] - my_vectors[i][2], 2)
                        );
                        if (d < minDistAny) minDistAny = d;
                    }
                    for (let i = 0; i < mutuals_vectors.length; ++i) {
                        let d = Math.sqrt(
                            Math.pow(v[0] - mutuals_vectors[i][0], 2) +
                            Math.pow(v[1] - mutuals_vectors[i][1], 2) +
                            Math.pow(v[2] - mutuals_vectors[i][2], 2)
                        );
                        if (d < minDistAny) minDistAny = d;
                    }
                    if (minDistAny > bestDist) {
                        bestDist = minDistAny;
                        bestMsg = fuzzed;
                        bestVec = v;
                    }
                    if (minDistAny >= minDist) {
                        break;
                    }
                }
                // Plot the best fuzzed message
                if (bestVec) {
                    let trace = {
                        x: [bestVec[0]], y: [bestVec[1]], z: [bestVec[2]],
                        mode: 'markers', type: 'scatter3d',
                        marker: { size: 10, color: 'orange' },
                        name: 'Fuzzed Post',
                        text: [`<b>Fuzzed Post</b><br>${bestMsg}`],
                        hovertemplate: '%{text}<extra></extra>'
                    };
                    Plotly.addTraces('plot', trace);
                    document.getElementById('newPostResults').innerHTML += `<br><b>Fuzzed farthest post:</b> <span>${bestMsg}</span> (min distance to any node: ${bestDist.toFixed(4)})`;
                }
            }

            // Compute the center and radius of the main sphere
            function getMainSphereInfo(my_vectors, mutuals_vectors) {
                let all = my_vectors.concat(mutuals_vectors);
                let n = all.length;
                let center = [0,0,0];
                for (let i = 0; i < n; ++i) {
                    center[0] += all[i][0];
                    center[1] += all[i][1];
                    center[2] += all[i][2];
                }
                center = center.map(x => x/n);
                let maxR = 0;
                for (let i = 0; i < n; ++i) {
                    let d = Math.sqrt(
                        Math.pow(all[i][0] - center[0], 2) +
                        Math.pow(all[i][1] - center[1], 2) +
                        Math.pow(all[i][2] - center[2], 2)
                    );
                    if (d > maxR) maxR = d;
                }
                return {center, radius: maxR};
            }

            // Fuzz and plot until outside the edge of the main sphere
            async function fuzzUntilOutsideSphere(msg, maxTries = 100) {
                let my_vectors = cachedVectors.my_vectors;
                let mutuals_vectors = cachedVectors.mutuals_vectors;
                let {center, radius} = getMainSphereInfo(my_vectors, mutuals_vectors);
                let bestMsg = msg, bestVec = null, bestDist = -Infinity;
                for (let fuzzLevel = 1; fuzzLevel <= maxTries; ++fuzzLevel) {
                    let fuzzed = fuzzMessage(msg, fuzzLevel);
                    let resp = await fetch('/vectorize_new_post', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: fuzzed })
                    });
                    let result = await resp.json();
                    let v = result.vector;
                    // Distance from center
                    let distFromCenter = Math.sqrt(
                        Math.pow(v[0] - center[0], 2) +
                        Math.pow(v[1] - center[1], 2) +
                        Math.pow(v[2] - center[2], 2)
                    );
                    if (distFromCenter > bestDist) {
                        bestDist = distFromCenter;
                        bestMsg = fuzzed;
                        bestVec = v;
                    }
                    if (distFromCenter > radius) {
                        break;
                    }
                }
                // Plot the best fuzzed message
                if (bestVec) {
                    let trace = {
                        x: [bestVec[0]], y: [bestVec[1]], z: [bestVec[2]],
                        mode: 'markers', type: 'scatter3d',
                        marker: { size: 12, color: 'orange', symbol: 'diamond' },
                        name: 'Fuzzed Outside',
                        text: [`<b>Fuzzed Outside</b><br>${bestMsg}`],
                        hovertemplate: '%{text}<extra></extra>'
                    };
                    Plotly.addTraces('plot', trace);
                    document.getElementById('newPostResults').innerHTML += `<br><b>Fuzzed outside sphere:</b> <span>${bestMsg}</span> (distance from center: ${bestDist.toFixed(4)}, sphere radius: ${radius.toFixed(4)})`;
                }
            }

            // Fuzz and plot until way outside the edge of the main sphere
            async function fuzzUntilWayOutsideSphere(msg, factor = 3, maxTries = 200) {
                let my_vectors = cachedVectors.my_vectors;
                let mutuals_vectors = cachedVectors.mutuals_vectors;
                let {center, radius} = getMainSphereInfo(my_vectors, mutuals_vectors);
                let targetRadius = radius * factor;
                let bestMsg = msg, bestVec = null, bestDist = -Infinity;
                for (let fuzzLevel = 1; fuzzLevel <= maxTries; ++fuzzLevel) {
                    let fuzzed = fuzzMessage(msg, fuzzLevel);
                    let resp = await fetch('/vectorize_new_post', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: fuzzed })
                    });
                    let result = await resp.json();
                    let v = result.vector;
                    // Distance from center
                    let distFromCenter = Math.sqrt(
                        Math.pow(v[0] - center[0], 2) +
                        Math.pow(v[1] - center[1], 2) +
                        Math.pow(v[2] - center[2], 2)
                    );
                    if (distFromCenter > bestDist) {
                        bestDist = distFromCenter;
                        bestMsg = fuzzed;
                        bestVec = v;
                    }
                    if (distFromCenter > targetRadius) {
                        break;
                    }
                }
                // Plot the best fuzzed message
                if (bestVec) {
                    let trace = {
                        x: [bestVec[0]], y: [bestVec[1]], z: [bestVec[2]],
                        mode: 'markers', type: 'scatter3d',
                        marker: { size: 14, color: 'magenta', symbol: 'star' },
                        name: 'Way Outside',
                        text: [`<b>Way Outside</b><br>${bestMsg}`],
                        hovertemplate: '%{text}<extra></extra>'
                    };
                    Plotly.addTraces('plot', trace);
                    document.getElementById('newPostResults').innerHTML += `<br><b>Way outside sphere:</b> <span>${bestMsg}</span> (distance from center: ${bestDist.toFixed(4)}, sphere radius: ${radius.toFixed(4)}, factor: ${factor})`;
                }
            }

            // Add button to fuzz message and plot outside sphere
            let fuzzBtn = document.createElement('button');
            fuzzBtn.textContent = 'Fuzz & Plot Outside Sphere';
            fuzzBtn.style.marginLeft = '1em';
            fuzzBtn.onclick = async function() {
                let postText = document.getElementById('newPostInput').value.trim();
                if (!postText) return;
                await fuzzUntilOutsideSphere(postText, 100);
            };
            document.getElementById('newPostForm').appendChild(fuzzBtn);

            // Add button to fuzz message and plot farthest
            let fuzzBtnFar = document.createElement('button');
            fuzzBtnFar.textContent = 'Fuzz & Plot Far';
            fuzzBtnFar.style.marginLeft = '1em';
            fuzzBtnFar.onclick = async function() {
                let postText = document.getElementById('newPostInput').value.trim();
                if (!postText) return;
                await fuzzUntilFar(postText, 10, 50);
            };
            document.getElementById('newPostForm').appendChild(fuzzBtnFar);

            // Add button to fuzz message and plot way outside sphere
            let fuzzBtnWayOutside = document.createElement('button');
            fuzzBtnWayOutside.textContent = 'Fuzz & Plot Way Outside Sphere';
            fuzzBtnWayOutside.style.marginLeft = '1em';
            fuzzBtnWayOutside.onclick = async function() {
                let postText = document.getElementById('newPostInput').value.trim();
                if (!postText) return;
                await fuzzUntilWayOutsideSphere(postText, 3, 200);
            };
            document.getElementById('newPostForm').appendChild(fuzzBtnWayOutside);
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
        posts = fetch_my_posts_with_uris()
        if not posts:
            print("No posts found. Debugging feed response...")
            feed = client.get_author_feed(BLUESKY_HANDLE)
            print("Raw feed:", feed)
        else:
            vectors = vectorize_posts(posts[0])
            print(f"Fetched {len(posts[0])} posts. First vector: {vectors[0]}")
