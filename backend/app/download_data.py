"""
Download 300+ public domain Wikipedia articles as sample corpus.
"""
import json
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")

TOPICS = [
    "Artificial intelligence", "Machine learning", "Deep learning", "Neural network",
    "Natural language processing", "Computer vision", "Robotics", "Data science",
    "Python programming language", "JavaScript", "React (software)", "FastAPI",
    "Database", "SQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Cloud computing", "Amazon Web Services", "Google Cloud Platform", "Microsoft Azure",
    "Docker (software)", "Kubernetes", "DevOps", "Continuous integration",
    "Cybersecurity", "Encryption", "Blockchain", "Bitcoin", "Ethereum",
    "Internet of Things", "Edge computing", "Quantum computing", "5G",
    "Climate change", "Renewable energy", "Solar panel", "Wind power",
    "Electric vehicle", "Tesla (company)", "SpaceX", "NASA",
    "Genome", "CRISPR", "Vaccine", "COVID-19 pandemic", "Epidemiology",
    "Psychology", "Cognitive science", "Neuroscience", "Behavioral economics",
    "Economics", "Macroeconomics", "Microeconomics", "Inflation", "Recession",
    "Stock market", "Venture capital", "Startup", "Entrepreneurship",
    "Marketing", "Brand management", "Supply chain", "Operations management",
    "Project management", "Agile software development", "Scrum (software development)",
    "Product management", "User experience", "User interface design",
    "Graphic design", "Typography", "Color theory",
    "History of the Internet", "World Wide Web", "Tim Berners-Lee",
    "Alan Turing", "Ada Lovelace", "Grace Hopper", "Linus Torvalds",
    "Elon Musk", "Jeff Bezos", "Bill Gates", "Steve Jobs",
    "Stanford University", "Massachusetts Institute of Technology", "Harvard University",
    "Oxford University", "Cambridge University",
    "United Nations", "World Health Organization", "World Trade Organization",
    "Philosophy", "Ethics", "Logic", "Epistemology",
    "Mathematics", "Statistics", "Linear algebra", "Calculus",
    "Physics", "Chemistry", "Biology", "Astronomy",
    "Albert Einstein", "Isaac Newton", "Charles Darwin", "Marie Curie",
    "Ancient Rome", "Ancient Greece", "Renaissance", "Industrial Revolution",
    "World War I", "World War II", "Cold War", "Space Race",
    "Democracy", "Capitalism", "Socialism", "Globalization",
    "Art", "Music", "Cinema", "Literature",
    "Sport", "Olympic Games", "Football", "Basketball",
    "Nutrition", "Exercise", "Mental health", "Meditation",
    "Architecture", "Urban planning", "Infrastructure", "Transportation",
    "Agriculture", "Food security", "Water resources", "Biodiversity",
    "Sustainability", "Circular economy", "Carbon footprint", "Paris Agreement",
    "International trade", "Foreign direct investment", "Monetary policy",
    "Central bank", "Federal Reserve", "European Union",
    "India", "China", "United States", "United Kingdom", "Germany",
    "Artificial neural network", "Reinforcement learning", "Transfer learning",
    "Computer science", "Software engineering", "Open source", "Linux"
]


def fetch_wikipedia_summary(title: str) -> dict | None:
    try:
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "KnowledgeSearchBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        extract = data.get("extract", "")
        if len(extract) < 100:
            return None
        return {
            "doc_id": f"wiki_{encoded.lower()}",
            "title": data.get("title", title),
            "text": extract,
            "source": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"  [skip] {title}: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "wikipedia_corpus.jsonl")

    if os.path.exists(out_path):
        with open(out_path) as f:
            count = sum(1 for _ in f)
        if count >= 300:
            print(f"Data already exists ({count} docs). Skipping download.")
            return

    print(f"Downloading Wikipedia articles to {out_path}...")
    docs = []
    for i, topic in enumerate(TOPICS):
        print(f"  [{i+1}/{len(TOPICS)}] {topic}")
        doc = fetch_wikipedia_summary(topic)
        if doc:
            docs.append(doc)
        time.sleep(0.3)

    with open(out_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"\nDownloaded {len(docs)} documents -> {out_path}")


if __name__ == "__main__":
    main()
