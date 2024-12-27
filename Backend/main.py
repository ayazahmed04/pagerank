from flask import Flask, request, render_template, jsonify
from bs4 import BeautifulSoup
import requests
import networkx as nx
import numpy as np

app = Flask(__name__)

def scrape_links(base_url):
    """
    Scrape internal links from the given website.
    """
    try:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for tag in soup.find_all("a", href=True):
            href = tag['href']
            if href.startswith(base_url) or not href.startswith("http"):
                full_url = requests.compat.urljoin(base_url, href)
                links.add(full_url)
        return list(links)
    except Exception as e:
        print(f"Error scraping {base_url}: {e}")
        return []

def build_adjacency_matrix(links):
    """
    Build adjacency matrix for PageRank.
    """
    link_to_index = {link: i for i, link in enumerate(links)}
    n = len(links)
    matrix = np.zeros((n, n))
    
    for i, link in enumerate(links):
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup.find_all("a", href=True):
                href = tag['href']
                full_url = requests.compat.urljoin(link, href)
                if full_url in link_to_index:
                    matrix[link_to_index[full_url], i] += 1
        except:
            continue
    
    return matrix, links

def compute_pagerank(matrix, damping_factor=0.85, max_iter=100, tol=1e-6):
    """
    Compute PageRank values.
    """
    n = matrix.shape[0]
    out_degree = matrix.sum(axis=0, keepdims=True)
    transition_matrix = np.divide(matrix, out_degree, where=out_degree != 0)
    ranks = np.ones(n) / n
    
    for _ in range(max_iter):
        new_ranks = (1 - damping_factor) / n + damping_factor * transition_matrix.dot(ranks)
        if np.linalg.norm(new_ranks - ranks, 1) < tol:
            break
        ranks = new_ranks
    
    return ranks

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        links = scrape_links(url)
        if not links:
            return render_template("index.html", error="Unable to scrape the website or no links found.")
        
        adjacency_matrix, link_list = build_adjacency_matrix(links)
        ranks = compute_pagerank(adjacency_matrix)
        ranked_links = sorted(zip(link_list, ranks), key=lambda x: x[1], reverse=True)
        
        return render_template("index.html", ranked_links=ranked_links)
    return render_template("index.html")

@app.route("/graph", methods=["POST"])
def graph():
    url = request.form.get("url")
    links = scrape_links(url)
    if not links:
        return jsonify({"error": "Unable to scrape the website or no links found."}), 400

    adjacency_matrix, link_list = build_adjacency_matrix(links)
    ranks = compute_pagerank(adjacency_matrix)

    # Create graph data
    graph = nx.DiGraph()
    for i, link in enumerate(link_list):
        for j, weight in enumerate(adjacency_matrix[:, i]):
            if weight > 0:
                graph.add_edge(link_list[i], link_list[j], weight=weight)

    positions = nx.spring_layout(graph)
    graph_data = {
        "nodes": [{"id": link, "rank": ranks[i]} for i, link in enumerate(link_list)],
        "edges": [{"source": u, "target": v, "weight": data["weight"]} for u, v, data in graph.edges(data=True)],
        "positions": {node: list(pos) for node, pos in positions.items()},
    }
    return jsonify(graph_data)

if __name__ == "__main__":
    app.run(debug=True)
