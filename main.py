from utils import analyze_narrative_frame, cache_get, generate_rag_response, load_articles_and_topic, analyze_articles, visualize_results, call_framing_api, generate_llm_queries, save_json, cache_set, run_visualizations_from_results
import json, os

MODEL_TYPE = "finetuned"
OUTPUT_DIR = "output_visualizations"
MODEL_NAME = "claude-sonnet-4.6"
CORPUS_DISTRIBUTION = "balanced"

def main():
    articles_and_topic = load_articles_and_topic("./final_corpora/final_articles.json")
    articles = articles_and_topic["articles"]
    topic = articles_and_topic["topic"]

    queries = generate_llm_queries(articles, topic)
    for idx, query in enumerate(queries):
        print(f"Generated LLM Query [{idx + 1}]:", query)

    rag_results = []
    rag_responses = []
    for idx, query in enumerate(queries):
        rag_response = generate_rag_response(articles, query)
        rag_responses.append(rag_response)

        cache_key = f"rag_result:{query}"
        cached_result = cache_get(cache_key, MODEL_TYPE)
        if cached_result is not None:
            print(f"  → Cache hit: RAG result for query [{idx + 1}]")
            rag_results.append(cached_result)
            continue

        rag_result = {"generic_framing": {}, "narrative_framing": {}}
        rag_result["generic_framing"] = call_framing_api(
            rag_response, label=f"RAG Output | query [{idx + 1}]", model_type=MODEL_TYPE
        )
        rag_result["narrative_framing"] = analyze_narrative_frame(
            rag_response, label=f"RAG Output | query [{idx + 1}]"
        )
        rag_result["query"] = idx

        if rag_result["generic_framing"] and rag_result["narrative_framing"]:
            cache_set(cache_key, MODEL_TYPE, rag_result)

        rag_results.append(rag_result)
    
    article_results = analyze_articles(articles, MODEL_TYPE, do_narrative=True)
    for idx, rag_result in enumerate(rag_results):
        visualize_results([x["generic_framing"] for x in article_results], topic, rag_result["generic_framing"], f"visualizations/{CORPUS_DISTRIBUTION}/query_{idx + 1}", MODEL_NAME)

    save_json(f"./results/{CORPUS_DISTRIBUTION}/{topic.lower().replace(' ', '-')}_{MODEL_NAME}_{MODEL_TYPE}_results.json", {
        "topic": topic,
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "article_results": article_results,
        "rag_results": rag_results,
        "queries": queries,
        "rag_responses": rag_responses
    })


# def analyze_baseline():
#     results = call_framing_api("Border \& Deportations Unauthorized border crossings have collapsed — from roughly 88,000 monthly encounters in early 2025 to just over 7,000 per month by late 2025. Migration Policy Institute The administration frames this as a success; critics point out it came paired with sweeping due process restrictions. For the first time in decades, the U.S. saw net negative migration in 2025 — more people left than entered. Mantra Law Office Interior Enforcement \& Detention By November 2025, ICE was using 91% more detention facilities than at the start of the year, including tent facilities on military bases holding up to 5,000 people and newly-constructed state-run facilities. American Immigration Council The controversy: critics argue the administration is primarily using detention to pressure people into abandoning their legal cases, rather than targeting serious public safety threats — with 14.3 people deported directly from ICE custody for every one released pending a hearing. American Immigration Council Legal \& Constitutional Disputes Recent memos allow ICE to enter homes using administrative rather than judicial warrants Mantra Law Office — a practice widely criticized by constitutional lawyers. The administration\'s enforcement operations have been accused of racial profiling and detaining citizens and noncitizens alike without due process. American Immigration Council States like Colorado, New Mexico, and Maryland have pushed back with laws limiting data-sharing and requiring federal court warrants for entry into government facilities. Refugee \& Asylum Policy Only 506 refugees were resettled from February to October 2025 — compared to over 100,000 in FY 2024 — with the FY 2026 ceiling set at a record-low 7,500, most of those reserved for White South Africans. Migration Policy Institute That prioritization has drawn pointed criticism as discriminatory. Visa \& Legal Immigration Starting January 21, 2026, immigrant visa issuances were paused indefinitely for nationals of roughly 75 countries Mantra Law Office, and work permit validity was cut from five years to as little as 18 months Mantra Law Office to allow more frequent vetting. The Core Tension The administration argues these are overdue security measures restoring integrity to a broken system. Opponents argue the scale, speed, and targeting of enforcement has moved well beyond legality into what amounts to mass punishment without adequate judicial oversight. Both sides can point to real evidence — which should tell you the debate is genuinely contested, not simply one side being right.", label="Baseline", model_type=MODEL_TYPE)
#     print(results)


def main_run_visualizations():
    run_visualizations_from_results(
        results_path="results/balanced/immigration-in-the-united-states_claude-sonnet-4.6_finetuned_results.json",
        out_dir="visualizations/balanced",
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    # main()
    # analyze_baseline()
    main_run_visualizations()