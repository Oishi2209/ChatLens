import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def parse_chat(file):
    raw_text = file.read().decode("utf-8")
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?[ap]m)\s-\s(.+?):\s(.+)"
    matches = re.findall(pattern, raw_text, re.IGNORECASE)
    df = pd.DataFrame(matches, columns=["date", "time", "sender", "message"])
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"],
                                      format="%d/%m/%y %I:%M %p")
    df = df.drop(columns=["date", "time"])
    return df

def get_sentiment(message):
    return analyzer.polarity_scores(message)["compound"]

def label_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def build_social_graph(df):
    G = nx.DiGraph()
    df = df.sort_values("timestamp").reset_index(drop=True)
    for i in range(1, len(df)):
        current_sender = df.loc[i, "sender"]
        previous_sender = df.loc[i - 1, "sender"]
        time_diff = (df.loc[i, "timestamp"] - df.loc[i-1, "timestamp"]).total_seconds()
        if current_sender == previous_sender:
            continue
        if time_diff <= 60:
            if G.has_edge(current_sender, previous_sender):
                G[current_sender][previous_sender]["weight"] += 1
            else:
                G.add_edge(current_sender, previous_sender, weight=1)
    return G

def get_topics(df, n_topics=4, n_words=6):
    df = df[df["message"].str.split().str.len() >= 3].copy()
    stop_words = ["media", "omitted", "message", "deleted", "edited",
                  "ok", "okay", "yeah", "yes", "haha", "lol", "im",
                  "ill", "ive", "dont", "cant", "wont", "ur", "u", "r",
                  "the", "a", "an", "is", "it", "in", "on", "and", "to",
                  "of", "for", "be", "was", "are", "this", "that", "with"]
    vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.9, min_df=2)
    message_matrix = vectorizer.fit_transform(df["message"])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(message_matrix)
    words = vectorizer.get_feature_names_out()
    topics = []
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[-n_words:]][::-1]
        topics.append(f"Topic {i+1}: {', '.join(top_words)}")
    return topics

#streamlit app

st.set_page_config(page_title="ChatLens", layout="wide")
st.title("ChatLens: Friend Group Dynamics Analyzer")
st.write("Upload a WhatsApp group chat export (.txt) to analyze your group's dynamics.")

uploaded_file = st.file_uploader("Upload chat.txt", type="txt")

if uploaded_file is not None:

    # parse
    df = parse_chat(uploaded_file)
    df["sentiment"] = df["message"].apply(get_sentiment)
    df["sentiment_label"] = df["sentiment"].apply(label_sentiment)

    st.success(f"Loaded {len(df)} messages from {df['sender'].nunique()} people")

    # Section 1: Raw data 
    st.header("Raw Messages")
    st.dataframe(df[["timestamp", "sender", "message", "sentiment_label"]].head(20))

    # Section 2: Sentiment 
    st.header("Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Sentiment Breakdown")
        sentiment_counts = df["sentiment_label"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index,
                autopct="%1.1f%%", colors=["#90EE90", "#FFB6C1", "#D3D3D3"])
        ax1.set_title("Sentiment Distribution")
        st.pyplot(fig1)

    with col2:
        st.subheader("Average Sentiment per Person")
        avg_sentiment = df.groupby("sender")["sentiment"].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        avg_sentiment.plot(kind="barh", ax=ax2, color="skyblue")
        ax2.set_xlabel("Average Sentiment Score")
        ax2.set_title("Who is most positive?")
        plt.tight_layout()
        st.pyplot(fig2)

    # Section 3: Activity 
    st.header("Activity Analysis")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Messages per Person")
        msg_counts = df["sender"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        msg_counts.plot(kind="barh", ax=ax3, color="coral")
        ax3.set_xlabel("Number of Messages")
        ax3.set_title("Who talks the most?")
        plt.tight_layout()
        st.pyplot(fig3)

    with col4:
        st.subheader("Activity Over Time")
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date").size()
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        daily.plot(ax=ax4, color="purple")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Messages")
        ax4.set_title("Daily Message Volume")
        plt.tight_layout()
        st.pyplot(fig4)

    # Section 4: Social Graph
    st.header("Social Network Graph")

    G = build_social_graph(df)
    pagerank = nx.pagerank(G)

    fig5, ax5 = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=3)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    node_sizes = [pagerank[node] * 10000 for node in G.nodes()]

    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 1]
    weights_to_draw = [G[u][v]["weight"] for u, v in edges_to_draw]

    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, 
                        width=[w * 0.5 for w in weights_to_draw],
                        arrows=True, edge_color="gray", ax=ax5)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                        node_color="skyblue", ax=ax5)
    nx.draw_networkx_labels(G, pos, font_size=5, ax=ax5)
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in weights],
                        arrows=True, edge_color="gray", ax=ax5)

    ax5.set_title("Who replies to whom?")
    ax5.axis("off")
    plt.tight_layout()
    st.pyplot(fig5)

    st.subheader("Influence Scores")
    scores = pd.DataFrame({
        "pagerank": pagerank,
        "degree_centrality": nx.degree_centrality(G)
    }).sort_values("pagerank", ascending=False)
    st.dataframe(scores)
    # Section 5: Topics
    st.header("Topic Modeling")
    topics = get_topics(df)
    for t in topics:
        st.write("•", t)

    st.success("Analysis complete!")