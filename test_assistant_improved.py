"""
test_assistant.py
-----------------
Evaluates the market research assistant using Precision, Recall, and F1 score.
Ground truth is based on manually reviewed retriever outputs.

Tests:
  Q1 - Input validation (valid vs invalid inputs)
  Q2 - URL relevance (F1 against manually defined ground truth)
  Q3 - Report faithfulness (cosine similarity between report and source pages)

Run with:
    python test_assistant.py
"""

import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "put_your_API_key_here"
os.environ["GOOGLE_API_KEY"] = API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=API_KEY)
retriever = WikipediaRetriever(top_k_results=5, lang="en")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)

# How long to wait between LLM calls to avoid rate limits (seconds)
SLEEP = 15

# ── Ground truth ──────────────────────────────────────────────────────────────
# Defined by manually running the assistant and reviewing each returned page.
# A page is RELEVANT if it covers the industry broadly.
# A page is IRRELEVANT if it is a specific company, person, or geographic variant.
#
# Observed results:
#   fintech  → 1/5 relevant (e.g. returned "Ozerk Ozan", "Jane Gladstone")
#   finance  → 1-2/5 relevant (e.g. returned "Ministry of Economics France")
#   aviation → 1/5 relevant (e.g. returned "Shahed Aviation Industries")
#   tech     → 2/5 relevant (e.g. returned "Tech bro", "Science and tech in Israel")

GROUND_TRUTH = {
    # Oracle set: ~15 genuinely relevant Wikipedia pages per industry.
    # Includes both industry overview pages AND major global companies.
    # Neither test system was designed to match this list specifically.
    # A retrieved page counts as TP if its title contains any keyword below.
    # Source: manually compiled from Wikipedia + industry knowledge.
    "fintech": {
        "relevant": [
            "financial technology",
            "digital banking",
            "mobile payment",
            "neobank",
            "open banking",
            "cryptocurrency",
            "blockchain",
            "digital wallet",
            "buy now pay later",
            "paypal",
            "visa",
            "mastercard",
            "stripe",
            "revolut",
            "klarna",
            "ant group",
            "adyen",
            "coinbase",
            "nubank",
            "wise",
        ],
    },
    "finance": {
        "relevant": [
            "finance",
            "financial services",
            "investment banking",
            "stock market",
            "capital market",
            "hedge fund",
            "private equity",
            "asset management",
            "retail banking",
            "jpmorgan",
            "goldman sachs",
            "morgan stanley",
            "blackrock",
            "bank of america",
            "citigroup",
            "hsbc",
            "barclays",
            "wells fargo",
            "ubs",
            "deutsche bank",
        ],
    },
    "aviation": {
        "relevant": [
            "aviation",
            "aerospace",
            "airline",
            "commercial aviation",
            "aircraft",
            "air transport",
            "history of aviation",
            "boeing",
            "airbus",
            "lockheed martin",
            "rolls-royce",
            "embraer",
            "general electric",
            "raytheon",
            "delta air",
            "united airlines",
            "american airlines",
            "lufthansa",
            "ryanair",
            "safran",
        ],
    },
    "tech": {
        "relevant": [
            "information technology",
            "technology company",
            "software industry",
            "semiconductor",
            "artificial intelligence",
            "cloud computing",
            "social media",
            "apple",
            "microsoft",
            "google",
            "alphabet",
            "amazon",
            "meta",
            "nvidia",
            "samsung",
            "intel",
            "ibm",
            "oracle",
            "salesforce",
            "tsmc",
        ],
    },
}


# ── Helper functions ──────────────────────────────────────────────────────────

def llm_call(prompt: str) -> str:
    """Wrapper that sleeps before every LLM call to respect rate limits."""
    time.sleep(SLEEP)
    response = llm.invoke(prompt)
    return response.content


def validate_industry(user_input: str) -> bool:
    prompt = f'Is "{user_input}" a valid industry or business sector? Reply with only YES or NO.'
    return llm_call(prompt).strip().upper().startswith("YES")


def get_retrieved_titles(industry: str) -> list[str]:
    # LLM suggests specific Wikipedia titles (mirrors current app approach)
    suggest_prompt = f"""Give me exactly 7 Wikipedia page titles for a research report on the {industry} industry.
Use this structure:
- 2 pages about the {industry} industry itself (e.g. overview, history, key concepts)
- 5 pages for the most famous, globally recognised companies in the {industry} industry (household names or market leaders with billions in revenue)

Rules:
- Use the exact, unambiguous Wikipedia page title (e.g. "Block, Inc." not "Square", "Visa Inc." not "Visa")
- Only include companies with a well-known, dedicated Wikipedia page
- Do not include niche, regional, or ambiguous company names
Reply with exactly 7 page titles, one per line, nothing else."""
    time.sleep(SLEEP)
    response = llm.invoke(suggest_prompt)
    suggested_titles = [t.strip().lstrip("0123456789.-) ") for t in response.content.strip().split("\n") if t.strip()][:7]

    docs = []
    seen_titles = set()
    for title in suggested_titles:
        if len(docs) >= 5:
            break
        try:
            results = retriever.invoke(title)
            if results:
                fetched_title = results[0].metadata.get('title', '')
                if fetched_title not in seen_titles:
                    seen_titles.add(fetched_title)
                    docs.append(results[0])
        except Exception:
            continue

    return [doc.metadata.get("title", "").lower() for doc in docs]


def generate_report(industry: str, docs) -> str:
    context = "\n\n".join([d.page_content[:800] for d in docs[:5]])
    prompt = f"""You are a market research analyst. Based only on the following Wikipedia extracts,
write a concise industry report on the {industry} industry in under 500 words.

Wikipedia content:
{context}

Report:"""
    return llm_call(prompt)


def compute_f1(retrieved_titles: list[str], relevant_keywords: list[str]) -> dict:
    tp = sum(1 for title in retrieved_titles if any(kw in title for kw in relevant_keywords))
    fp = len(retrieved_titles) - tp
    fn = sum(1 for kw in relevant_keywords if not any(kw in title for title in retrieved_titles))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
    }


# ── Q1: Input validation ──────────────────────────────────────────────────────

def test_q1():
    print("\n" + "="*60)
    print("Q1 — Input Validation")
    print("="*60)
    print(f"  (sleeping {SLEEP}s between calls to respect rate limits)\n")

    valid_inputs   = ["fintech", "finance", "aviation", "tech"]
    invalid_inputs = ["", "42", "!!!", "asdfghjkl"]  # reduced to save quota

    correct = 0
    total   = len(valid_inputs) + len(invalid_inputs)

    for inp in valid_inputs:
        result = validate_industry(inp)
        status = "✓" if result else "✗"
        print(f"  [{status}] '{inp}' → {'VALID' if result else 'INVALID (wrong!)'}")
        if result:
            correct += 1

    for inp in invalid_inputs:
        result = validate_industry(inp)
        status = "✓" if not result else "✗"
        print(f"  [{status}] '{inp}' → {'INVALID' if not result else 'VALID (wrong!)'}")
        if not result:
            correct += 1

    accuracy = correct / total
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.0%}")
    return accuracy


# ── Q2: URL relevance (F1) ────────────────────────────────────────────────────

def test_q2():
    print("\n" + "="*60)
    print("Q2 — URL Relevance (Precision / Recall / F1)")
    print("="*60)
    print("  Ground truth defined by manually reviewing assistant outputs.\n")

    all_f1 = []

    for industry, gt in GROUND_TRUTH.items():
        titles = get_retrieved_titles(industry)
        relevant_keywords = [kw.lower() for kw in gt["relevant"]]
        scores = compute_f1(titles, relevant_keywords)

        print(f"  Industry: {industry}")
        print(f"  Retrieved titles:")
        for t in titles:
            is_rel = any(kw in t for kw in relevant_keywords)
            print(f"    {'✓' if is_rel else '✗'} {t}")
        print(f"  TP:{scores['tp']}  FP:{scores['fp']}  FN:{scores['fn']}")
        print(f"  Precision: {scores['precision']}  Recall: {scores['recall']}  F1: {scores['f1']}\n")

        all_f1.append(scores["f1"])

    avg_f1 = round(sum(all_f1) / len(all_f1), 2)
    print(f"  Average F1 across all industries: {avg_f1}")
    return avg_f1


# ── Q3: Report faithfulness (cosine similarity) ───────────────────────────────

def test_q3():
    print("\n" + "="*60)
    print("Q3 — Report Faithfulness (Cosine Similarity)")
    print("="*60)
    print("  Method: embed report + source pages using Gemini embeddings,")
    print("  compute cosine similarity. Higher = more grounded in sources.\n")

    all_scores = []

    for industry in GROUND_TRUTH.keys():
        print(f"  Generating report for: {industry}...")
        # Use same LLM-guided retrieval as the app
        suggest_prompt = f"""Give me exactly 7 Wikipedia page titles for a research report on the {industry} industry.
- 2 pages about the {industry} industry itself
- 5 pages for the most famous globally recognised companies in the {industry} industry
Use exact, unambiguous Wikipedia page titles. Reply with exactly 7 titles, one per line, nothing else."""
        time.sleep(SLEEP)
        resp = llm.invoke(suggest_prompt)
        suggested = [t.strip().lstrip("0123456789.-) ") for t in resp.content.strip().split("\n") if t.strip()][:7]
        docs = []
        seen = set()
        for t in suggested:
            if len(docs) >= 5:
                break
            try:
                res = retriever.invoke(t)
                if res and res[0].metadata.get('title','') not in seen:
                    seen.add(res[0].metadata.get('title',''))
                    docs.append(res[0])
            except Exception:
                continue
        report = generate_report(industry, docs)

        source_texts = [d.page_content[:800] for d in docs[:5]]
        report_emb   = embeddings.embed_query(report)
        source_embs  = embeddings.embed_documents(source_texts)

        sims     = [cosine_similarity([report_emb], [s])[0][0] for s in source_embs]
        avg_sim  = round(float(np.mean(sims)), 3)
        word_count = len(report.split())

        print(f"  Industry       : {industry}")
        print(f"  Cosine sims    : {[round(s, 3) for s in sims]}")
        print(f"  Average sim    : {avg_sim}")
        print(f"  Word count     : {word_count} {'✓' if word_count < 500 else '✗ EXCEEDS 500'}\n")

        all_scores.append(avg_sim)

    avg_cos = round(float(np.mean(all_scores)), 3)
    print(f"  Average cosine similarity across industries: {avg_cos}")
    return avg_cos


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧪 Market Research Assistant — Evaluation Suite")
    print("   Industries tested: fintech, finance, aviation, tech")
    print(f"   Note: {SLEEP}s sleep between LLM calls — this will take ~10 mins total\n")

    q1_score = test_q1()

    print("\n  Pausing 60s before Q2 to reset rate limit...")
    time.sleep(60)

    q2_score = test_q2()

    print("\n  Pausing 60s before Q3 to reset rate limit...")
    time.sleep(60)

    q3_score = test_q3()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Q1 Validation accuracy          : {q1_score:.0%}")
    print(f"  Q2 URL relevance avg F1         : {q2_score:.2f}")
    print(f"  Q3 Report faithfulness avg cos  : {q3_score:.3f}")
    print("="*60)
