import streamlit as st
import time
from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png", width=200)

st.markdown("<h1 style='text-align: center;'>Wikipedia Retriever</h1>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("Google Gemini API Key:", type="password")

    st.divider()
    st.subheader("Model Parameters")

    model = st.selectbox(
    "Model:",
    ["gemini-2.5-flash"],
    help="Gemini 2.5 Flash"
)
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher = more creative, lower = more focused and deterministic"
    )

# ── About expander ────────────────────────────────────────────────────────────
with st.expander("About Wikipedia Retriever"):
    st.write("""
    **Wikipedia Industry Retriever**
    
    This tool helps you research industries by:
    1. Entering an industry name
    2. Retrieving 5 relevant Wikipedia pages
    3. Generating a professional industry report
    
    **How to use:**
    - Enter your Google Gemini API key in the sidebar
    - Select a model and adjust the temperature if desired
    - Type an industry (e.g., "Aviation", "Healthcare") and hit Search
    """)

# ── Main form ─────────────────────────────────────────────────────────────────
with st.form(key="industry_form"):
    industry = st.text_input("Please enter your industry of choice here:")
    submitted = st.form_submit_button("Search")

if submitted and industry:
    if not api_key:
        st.error("Please enter your Google Gemini API key in the sidebar to continue.")
        st.stop()

    # Step 1: Validate input
    with st.spinner("Validating input..."):
        try:
            validation_llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
            validation = validation_llm.invoke(f'Is "{industry}" a valid industry or business sector? Reply YES or NO only.')
        except Exception as e:
            st.error(f"Error connecting to Gemini: {e}")
            st.stop()

    if "NO" in validation.content.upper():
        st.error("That doesn't appear to be a valid industry. Please enter a valid industry name (e.g., 'Aviation', 'Healthcare').")
        st.stop()

    # Step 2: Get Wikipedia pages
    st.write("---")
    start_time = time.time()

    # Ask LLM for industry overview pages + top companies
    with st.spinner("Identifying relevant Wikipedia pages..."):
        suggest_llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
        suggest_prompt = f"""Give me exactly 7 Wikipedia page titles for a research report on the {industry} industry.
Use this structure:
- 2 pages about the {industry} industry itself (e.g. overview, history, key concepts)
- 5 pages for the most famous, globally recognised companies in the {industry} industry (think household names or market leaders with billions in revenue — not niche or regional players)

Only include companies that definitely have their own well-known Wikipedia page.
Reply with exactly 7 page titles, one per line, nothing else."""
        suggest_response = suggest_llm.invoke(suggest_prompt)
        suggested_titles = [t.strip().lstrip("0123456789.-) ") for t in suggest_response.content.strip().split("\n") if t.strip()][:7]

    # Fetch each suggested title directly, deduplicate by title, stop at 5
    with st.spinner("Fetching Wiki Pages..."):
        retriever = WikipediaRetriever(top_k_results=1, doc_content_chars_max=1000)
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

    fetch_time = time.time() - start_time

    st.write("### Top 5 Wikipedia Pages:")
    for doc in docs:
        url = doc.metadata.get('source', 'No URL')
        title = doc.metadata.get('title', 'Wikipedia Page')
        st.markdown(f"🔗 [{title}]({url})")
    st.write(f"Fetch time: {fetch_time:.2f} seconds")

    # Step 3: Generate report
    st.write("---")
    st.write("### Industry Report:")

    start_time = time.time()
    with st.spinner("Writing report..."):
        combined_content = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Based on the following Wikipedia content about {industry}, write a professional industry report.

        The report must be less than 500 words and include key insights about the industry.

        Wikipedia Content:
        {combined_content}

        Industry Report:"""

        report_llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)
        response = report_llm.invoke(prompt)

    report_time = time.time() - start_time

    report = response.content
    st.markdown(report.replace('$', '\\$'))

    word_count = len(report.split())
    st.write(f"**Word count:** {word_count}/500")
    st.write(f"Model: **{model}** | Temperature: {temperature} | Time: {report_time:.2f}s")

elif submitted and not industry:
    st.warning("Please enter an industry to continue.")
