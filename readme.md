# ChatLens: Group Chat Dynamics Analyzer

ChatLens is a Streamlit dashboard that analyzes a WhatsApp group chat export. Upload the `.txt` file and it shows how positive each member's messages are, who replies to whom, who is most influential in the group, and what topics the group discusses most.

![ChatLens dashboard](docs/dashboard.gif)

## Features
- Parses raw WhatsApp exports into a timestamped table
- Per-message sentiment scoring with VADER, with averages per member
- Directed reply graph built with NetworkX
- Influence ranking with PageRank and degree centrality
- Topic discovery with Latent Dirichlet Allocation (LDA)
- Interactive, multi-section dashboard built with Streamlit

## Tech Stack
Python · Streamlit · Pandas · NetworkX · scikit-learn · VADER Sentiment · Matplotlib · Regex

## Getting Started

### Prerequisites
- Python 3.10+

### Installation
```bash
git clone https://github.com/[your-username]/chatlens.git
cd chatlens
pip install -r requirements.txt
```

### Exporting Your WhatsApp Chat
1. Open the group chat.
2. **Android:** tap ⋮ → More → Export chat. **iPhone:** tap the group name → Export Chat.
3. Choose **Without Media**.
4. Save the resulting `.txt` file.

> Chat exports contain personal messages and names. Don't commit real exports to this repo. Test with a synthetic sample instead: [`data/sample_chat.txt`](data/sample_chat.txt).

### Usage
```bash
streamlit run app.py
```
Then upload your exported `.txt` file in the browser.

## How It Works

1. **Parse:** A regex extracts the date, time, sender, and message from each line. The date and time are combined into a single timestamp.
2. **Sentiment:** VADER gives each message a compound score from -1 to +1. Scores of +0.05 or higher are labeled positive, -0.05 or lower negative, and the rest neutral.
3. **Reply graph:** Messages are sorted by time. When a member sends a message within 60 seconds of a message from a different member, that counts as a reply. The directed edge points from the replier to the person being replied to, and its weight counts how many times this happened.
4. **Influence:** PageRank and degree centrality are computed on the graph to rank members.
5. **Topics:** Messages with at least three words are converted to a bag-of-words matrix, with English and chat-specific stopwords removed. LDA then finds four topics, each described by its six most representative words.

## Dashboard Sections
- **Raw Messages:** the first 20 parsed messages, with their sentiment labels
- **Sentiment Analysis:** the overall positive/negative/neutral split and the average sentiment per member
- **Activity Analysis:** message counts per member and daily message volume
- **Social Network Graph:** the reply network, with node size scaled by PageRank
- **Influence Scores:** a table of PageRank and degree centrality per member
- **Topic Modeling:** the LDA topics for the chat

## Limitations
- **Date format:** Only day/month/year dates are supported, which is the format most non-US WhatsApp exports use. US-style `m/d/yy` exports won't parse correctly.
- **Timestamp format:** Only 12-hour times with am/pm are recognized. 24-hour exports won't match the regex.
- **Multi-line messages:** Only the first line of a multi-line message is captured. Continuation lines are dropped.
- **Sentiment:** VADER is designed for English. Other languages and heavy slang are scored less reliably.
- **Reply detection:** Replies are inferred from timing, not from WhatsApp's quote feature, so the graph is an approximation.
- **Topics:** LDA produces word clusters that need human interpretation. The results depend on the chat's size and the number of topics.

## Future Improvements
- Support US-style dates and 24-hour timestamps
- Capture multi-line messages
- Make the reply window and the number of topics adjustable in the sidebar
- Export results to CSV

## Project Structure
```
chatlens/
├── data/
│   └── sample_chat.txt
├── docs/
│   └── dashboard.gif
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## License
MIT
