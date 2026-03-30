import mediacloud.api
from datetime import date
import trafilatura
import json, time
import matplotlib.pyplot as plt
from utils import analyze_articles

MODEL_TYPE = "finetuned"
API_KEY = "21f09a90445d444631abdab752a80ce77e781d2f"
US_NATIONAL_COLLECTION = 34412234

EXCLUDED_KEYWORDS = ["tsa"]
REQUIRED_KEYWORDS = ["immigration", "immigrant", "migrants", "migration"]
OPTIONAL_KEYWORDS = ["trump", "ice", "dhs", "border", "deport", "clergy", "church", "enforcement", "federal", "minnesota", "noem", "illegal", "agents", "administration", "pretti", "renee", "fraud", "economy"]
QUERY = "(immigration OR immigrant OR migrants OR migration) AND (trump OR ice OR dhs OR border OR deport OR clergy OR church OR enforcement OR federal OR minnesota OR noem OR illegal OR agents OR administration OR pretti OR renee OR fraud OR economy)"
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 3, 25)

MAX_STORIES = 1000
SLEEP_BETWEEN_REQUESTS = 1

mc = mediacloud.api.SearchApi(API_KEY)

def fetch_all_stories():
    all_stories = []
    pagination_token = None

    print("Fetching stories from MediaCloud...")

    while True:
        page, pagination_token = mc.story_list(
            QUERY,
            start_date=START_DATE,
            end_date=END_DATE,
            collection_ids=[US_NATIONAL_COLLECTION],
            pagination_token=pagination_token
        )

        all_stories.extend(page)

        print(f"Fetched {len(all_stories)} stories so far...")

        if pagination_token is None or len(all_stories) >= MAX_STORIES:
            break
    return all_stories

def extract_article_text(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None

        text = trafilatura.extract(downloaded)
        return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def main(existing=None, all_stories=[]):
    print("Fetching full article text...")

    if existing:
        results = existing
    else:
        results = []
    print("Existing:", len(results))

    for story in all_stories:
        url = story.get("url")
        title = story.get("title")

        if not url:
            continue

        print(f"Processing: {title}")

        article_text = extract_article_text(url)

        results.append({
            "title": title,
            "url": url,
            "text": article_text
        })

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    with open("news_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(results)} articles to news_results.json")

def dedup_articles(articles):
    deduped = {}
    for article in articles:
        title = article["title"].lower()
        if title in deduped:
            continue
        deduped[title] = article
        
    return deduped.values()

def compute_frame_counts(articles):
    counts = {"Co": 0, "Ec": 0, "Hi": 0, "Mo": 0, "Re": 0}
    for a in articles:
        print(a)
        g = a["generic_framing"]
        if g["conflict"]["present"]:      counts["Co"] += 1
        if g["economic"]["present"]:      counts["Ec"] += 1
        if g["human_interest"]["present"]: counts["Hi"] += 1
        if g["morality"]["present"]:      counts["Mo"] += 1
        if g["responsibility"]["present"]: counts["Re"] += 1
    return counts

def article_frames(article):
    g = article["generic_framing"]
    return {
        "Co": g["conflict"]["present"],
        "Ec": g["economic"]["present"],
        "Hi": g["human_interest"]["present"],
        "Mo": g["morality"]["present"],
        "Re": g["responsibility"]["present"],
    }

def imbalance_score(counts):
    vals = list(counts.values())
    return max(vals) - min(vals)

def greedy_prune(frame_results, target=None):
    articles = list(frame_results)  # copy
    
    while True:
        counts = compute_frame_counts(articles)
        if target is None:
            target = min(counts.values())
        
        # Check if we've reached balance
        if imbalance_score(counts) == 0:
            break
        
        # Find the most overrepresented frame
        max_frame = max(counts, key=counts.get)
        
        # If already at target, stop
        if counts[max_frame] <= target:
            break
        
        # Find the best article to remove:
        # - Must contain the overrepresented frame
        # - Should minimize damage to underrepresented frames
        min_count = min(counts.values())
        
        best_article = None
        best_score = float('-inf')
        
        for i, article in enumerate(articles):
            frames = article_frames(article)
            if not frames[max_frame]:
                continue  # only remove articles hurting the max frame
            
            # Penalize removing articles that contribute to underrepresented frames
            score = 0
            for frame, present in frames.items():
                if present:
                    # Removing this hurts frames proportional to how underrepresented they are
                    score -= (counts[max_frame] - counts[frame])  # less penalty if frame is also over-represented
            
            if score > best_score:
                best_score = score
                best_article = i
        
        if best_article is None:
            break
            
        articles.pop(best_article)
    
    return articles

def filter_articles_by_keywords(articles, required_keywords, optional_keywords, excluded_keywords, num_optional_keywords_required=1):
    filtered = []
    for article in articles:
        title = article.get("title") or ""
        text = article.get("text") or ""
        combined = (title + " " + text).lower()
        
        if any(kw in combined for kw in excluded_keywords):
            continue
        
        if not all(kw in combined for kw in required_keywords):
            continue
        
        optional_count = sum(1 for kw in optional_keywords if kw in combined)
        if optional_count < num_optional_keywords_required:
            continue
        
        filtered.append(article)
    
    return filtered

def balance_framing():
    with open("./final_corpora/post_excluded_tsa/filtered_corporus_frame_results.json") as f:
        frame_results = json.load(f)

    print("Before pruning:", compute_frame_counts(frame_results))
    pruned = greedy_prune(frame_results, 10)
    print("After pruning:", compute_frame_counts(pruned))
    print(f"Kept {len(pruned)}/{len(frame_results)} articles")

    with open("./final_corpora/post_excluded_tsa/filtered_corporus_frame_results_balanced.json", "w", encoding="utf-8") as f:
        json.dump(pruned, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    with open("final_corpora/deduped_results.json") as f:
        news_articles = json.load(f)
        
    filtered = filter_articles_by_keywords(
        news_articles, 
        required_keywords=REQUIRED_KEYWORDS, 
        optional_keywords=OPTIONAL_KEYWORDS, 
        excluded_keywords=EXCLUDED_KEYWORDS,
        num_optional_keywords_required=5
    )
    print(len(filtered), "articles after keyword filtering")

    _formatted = []
    for article in filtered:
        _formatted.append({
            "title": article["title"],
            "url": article["url"],
            "content": article["text"]
        })
    
    frame_results = analyze_articles(_formatted, MODEL_TYPE, do_narrative=False)
    frame_counts = compute_frame_counts(frame_results)
    print(frame_counts)

    with open("./final_corpora/post_excluded_tsa/filtered_corporus.json", "w", encoding="utf-8") as f:
        json.dump(_formatted, f, ensure_ascii=False, indent=4)

    with open("./final_corpora/post_excluded_tsa/filtered_corporus_frame_results.json", "w", encoding="utf-8") as f:
        json.dump(frame_results, f, ensure_ascii=False, indent=4)

    balance_framing()