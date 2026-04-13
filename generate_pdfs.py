"""
generate_pdfs.py
Run this once to create the 3 sample PDFs used by the RAG system.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import os

OUTPUT_DIR = "data/raw_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
styles = getSampleStyleSheet()
H1 = styles["Heading1"]
H2 = styles["Heading2"]
NM = styles["Normal"]
SP = lambda n=12: Spacer(1, n)


# ─────────────────────────────────────────────────────────────────
# PDF 1 : Introduction to Machine Learning
# ─────────────────────────────────────────────────────────────────
def make_pdf1():
    path = os.path.join(OUTPUT_DIR, "machine_learning_intro.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = []

    story.append(Paragraph("Introduction to Machine Learning", H1))
    story.append(SP())
    story.append(Paragraph(
        "Machine learning (ML) is a subset of artificial intelligence that enables systems "
        "to learn and improve from experience without being explicitly programmed. "
        "It focuses on developing computer programs that can access data and use it to learn for themselves.",
        NM))
    story.append(SP())

    story.append(Paragraph("1. Types of Machine Learning", H2))
    story.append(Paragraph(
        "Supervised Learning: The algorithm learns from labeled training data. "
        "Each training example consists of an input and the desired output. "
        "Common algorithms include Linear Regression, Decision Trees, Support Vector Machines, and Neural Networks. "
        "Applications include spam detection, image classification, and price prediction.",
        NM))
    story.append(SP(8))
    story.append(Paragraph(
        "Unsupervised Learning: The algorithm discovers hidden patterns in data without labels. "
        "Clustering algorithms like K-Means group similar data points together. "
        "Dimensionality reduction techniques like PCA reduce the number of features. "
        "Applications include customer segmentation and anomaly detection.",
        NM))
    story.append(SP(8))
    story.append(Paragraph(
        "Reinforcement Learning: An agent learns by interacting with an environment. "
        "The agent receives rewards for correct actions and penalties for incorrect ones. "
        "Famous examples include AlphaGo defeating world Go champions and autonomous vehicle control.",
        NM))
    story.append(SP())

    story.append(Paragraph("2. Key Concepts", H2))
    for item in [
        ("Training Data", "The dataset used to teach the model. Quality and quantity directly affect performance."),
        ("Features", "Individual measurable properties of the data used as inputs to the model."),
        ("Labels", "The output variable the model is trained to predict in supervised learning."),
        ("Overfitting", "When a model learns training data too well including its noise, performing poorly on new data."),
        ("Underfitting", "When a model is too simple to capture the underlying pattern in the data."),
        ("Cross-Validation", "A technique to evaluate model performance by splitting data into training and test sets."),
        ("Gradient Descent", "An optimization algorithm used to minimize the loss function during training."),
        ("Hyperparameters", "Configuration settings of the model that are set before training begins."),
    ]:
        story.append(Paragraph(f"<b>{item[0]}:</b> {item[1]}", NM))
        story.append(SP(6))

    story.append(Paragraph("3. Neural Networks and Deep Learning", H2))
    story.append(Paragraph(
        "Deep learning is a subset of machine learning that uses multi-layered neural networks. "
        "A neural network consists of an input layer, one or more hidden layers, and an output layer. "
        "Each neuron applies a weighted sum and an activation function such as ReLU or sigmoid. "
        "Backpropagation is used to update weights by computing gradients of the loss function. "
        "Convolutional Neural Networks (CNNs) excel at image tasks. "
        "Recurrent Neural Networks (RNNs) are designed for sequential data like text and time series. "
        "Transformers, introduced in 2017, revolutionized NLP through the self-attention mechanism.",
        NM))
    story.append(SP())

    story.append(Paragraph("4. Evaluation Metrics", H2))
    data = [["Metric", "Formula", "Use Case"],
            ["Accuracy", "Correct / Total", "Balanced classification"],
            ["Precision", "TP / (TP+FP)", "Minimize false positives"],
            ["Recall", "TP / (TP+FN)", "Minimize false negatives"],
            ["F1 Score", "2*(P*R)/(P+R)", "Imbalanced datasets"],
            ["MSE", "mean((y-y_hat)**2)", "Regression tasks"],
            ["R-squared", "1 - SS_res/SS_tot", "Regression fit quality"]]
    t = Table(data, colWidths=[1.5*inch, 2*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4A90D9")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F0F4FF"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(SP())

    story.append(Paragraph("5. Python Libraries for Machine Learning", H2))
    story.append(Paragraph(
        "Scikit-learn provides simple and efficient tools for data mining and data analysis. "
        "TensorFlow is an open-source platform for machine learning developed by Google. "
        "PyTorch is a deep learning framework developed by Facebook's AI Research lab. "
        "Pandas is used for data manipulation and analysis. "
        "NumPy provides support for large multi-dimensional arrays and matrices. "
        "Matplotlib and Seaborn are used for data visualization.",
        NM))

    doc.build(story)
    print(f"Created: {path}")


# ─────────────────────────────────────────────────────────────────
# PDF 2 : Retrieval-Augmented Generation (RAG)
# ─────────────────────────────────────────────────────────────────
def make_pdf2():
    path = os.path.join(OUTPUT_DIR, "rag_systems_guide.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = []

    story.append(Paragraph("Retrieval-Augmented Generation (RAG): A Comprehensive Guide", H1))
    story.append(SP())
    story.append(Paragraph(
        "Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of "
        "retrieval-based and generation-based approaches. RAG enhances large language models by "
        "providing them with relevant external knowledge retrieved from a document store, "
        "significantly reducing hallucinations and improving factual accuracy.",
        NM))
    story.append(SP())

    story.append(Paragraph("1. RAG Architecture Overview", H2))
    story.append(Paragraph(
        "A RAG system consists of two main components: the Retriever and the Generator. "
        "The Retriever finds relevant documents from a knowledge base using semantic similarity. "
        "The Generator (an LLM) produces answers conditioned on both the query and retrieved documents. "
        "The knowledge base stores documents as vector embeddings in a vector database such as FAISS or ChromaDB.",
        NM))
    story.append(SP())

    story.append(Paragraph("2. Indexing Pipeline", H2))
    for step, desc in [
        ("Document Loading", "Load raw documents from PDFs, websites, databases, or APIs."),
        ("Text Splitting", "Divide documents into manageable chunks (typically 256-1024 tokens) with overlap."),
        ("Embedding Generation", "Convert text chunks into dense vector representations using embedding models."),
        ("Vector Storage", "Store embeddings in a vector database for fast approximate nearest-neighbor search."),
    ]:
        story.append(Paragraph(f"<b>Step: {step}</b> — {desc}", NM))
        story.append(SP(6))
    story.append(SP())

    story.append(Paragraph("3. Query Pipeline", H2))
    story.append(Paragraph(
        "When a user submits a query, the system first converts the query into an embedding vector. "
        "The retriever performs a similarity search against the vector database to find the top-k most relevant chunks. "
        "These chunks are concatenated and passed to the LLM as context along with the original query. "
        "The LLM generates a response grounded in the retrieved context. "
        "The default similarity metric is cosine similarity, though Euclidean distance is also used.",
        NM))
    story.append(SP())

    story.append(Paragraph("4. Embedding Models", H2))
    data = [["Model", "Dimensions", "Speed", "Quality"],
            ["all-MiniLM-L6-v2", "384", "Very Fast", "Good — best for CPU"],
            ["all-mpnet-base-v2", "768", "Fast", "Very Good"],
            ["text-embedding-3-small", "1536", "API only", "Excellent"],
            ["bge-large-en-v1.5", "1024", "Medium", "Excellent"]]
    t = Table(data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F1F8E9"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(SP())

    story.append(Paragraph("5. Vector Databases", H2))
    story.append(Paragraph(
        "FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search. "
        "It supports CPU and GPU operation and can handle millions of vectors. "
        "ChromaDB is an open-source embedding database designed for AI applications. "
        "It provides persistence, filtering, and metadata support out of the box. "
        "Pinecone and Weaviate are cloud-hosted alternatives suited for production deployments.",
        NM))
    story.append(SP())

    story.append(Paragraph("6. RAG Evaluation Metrics", H2))
    for metric, desc in [
        ("Faithfulness", "Measures whether the answer is grounded in the retrieved context. Score 0-1."),
        ("Answer Relevancy", "Measures how relevant the answer is to the original question. Score 0-1."),
        ("Context Recall", "Measures how much of the ground truth is covered by the retrieved context."),
        ("Context Precision", "Measures whether retrieved chunks are actually relevant to the query."),
        ("Answer Correctness", "Compares generated answer to ground-truth using semantic similarity."),
    ]:
        story.append(Paragraph(f"<b>{metric}:</b> {desc}", NM))
        story.append(SP(6))
    story.append(SP())

    story.append(Paragraph("7. Advanced RAG Techniques", H2))
    story.append(Paragraph(
        "HyDE (Hypothetical Document Embeddings) generates a hypothetical answer and embeds it for retrieval. "
        "Multi-Query Retrieval generates multiple query variants to improve recall. "
        "Re-ranking applies a cross-encoder after initial retrieval to reorder results by relevance. "
        "Parent Document Retrieval stores large chunks but retrieves smaller child chunks. "
        "Self-RAG uses the LLM itself to decide when and what to retrieve.",
        NM))

    doc.build(story)
    print(f"Created: {path}")


# ─────────────────────────────────────────────────────────────────
# PDF 3 : Data Poisoning in AI Systems
# ─────────────────────────────────────────────────────────────────
def make_pdf3():
    path = os.path.join(OUTPUT_DIR, "data_poisoning_attacks.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = []

    story.append(Paragraph("Data Poisoning Attacks in AI and RAG Systems", H1))
    story.append(SP())
    story.append(Paragraph(
        "Data poisoning is a type of adversarial attack in which an attacker intentionally corrupts "
        "the training data or knowledge base of a machine learning system to degrade its performance "
        "or manipulate its outputs. In RAG systems, poisoning targets the document store rather than "
        "model weights, making it a particularly relevant and novel attack surface.",
        NM))
    story.append(SP())

    story.append(Paragraph("1. Categories of Poisoning Attacks", H2))
    for name, desc in [
        ("Label Flipping", "Correct content is replaced with factually opposite information. For example, replacing 'treatment increases survival' with 'treatment decreases survival'."),
        ("Backdoor Attacks", "A trigger phrase is embedded in malicious documents. When the trigger appears in a query, the system returns attacker-controlled content."),
        ("Noise Injection", "Irrelevant or random documents are added to the knowledge base to degrade retrieval precision and pollute context windows."),
        ("Semantic Poisoning", "Documents are subtly rephrased to change meaning while preserving surface-level fluency, making detection harder."),
        ("Sponge Attacks", "Computationally expensive documents are injected to slow down retrieval and exhaust system resources."),
    ]:
        story.append(Paragraph(f"<b>{name}:</b> {desc}", NM))
        story.append(SP(8))
    story.append(SP())

    story.append(Paragraph("2. Impact on RAG Systems", H2))
    story.append(Paragraph(
        "Poisoned RAG systems can produce factually incorrect answers with high confidence. "
        "The LLM may be misled by plausible-looking but false retrieved context. "
        "Backdoor attacks can cause the system to leak sensitive information when triggered. "
        "Noise injection degrades faithfulness and answer relevancy scores significantly. "
        "Unlike model poisoning, RAG poisoning does not require access to the model itself, "
        "only to the document ingestion pipeline.",
        NM))
    story.append(SP())

    story.append(Paragraph("3. Detection Techniques", H2))
    for tech, desc in [
        ("Provenance Tracking", "Record the source of every chunk. Flag answers derived from unverified or recently added sources."),
        ("Confidence Scoring", "Use embedding similarity between the query and retrieved chunks to detect low-relevance retrievals."),
        ("Semantic Consistency", "Cross-check generated answers against multiple retrieved chunks for contradictions."),
        ("Perplexity Filtering", "High-perplexity chunks relative to the document collection may indicate injected noise."),
        ("Majority Voting", "Retrieve from multiple independent stores and accept only answers consistent across all stores."),
    ]:
        story.append(Paragraph(f"<b>{tech}:</b> {desc}", NM))
        story.append(SP(6))
    story.append(SP())

    story.append(Paragraph("4. Defense Strategies", H2))
    story.append(Paragraph(
        "Input sanitization should be applied to all documents before ingestion. "
        "Cryptographic hashing can verify document integrity at retrieval time. "
        "Outlier detection identifies chunks whose embeddings are distant from the cluster centroid. "
        "Human-in-the-loop review is recommended for high-stakes domains such as healthcare and law. "
        "Differential privacy can be applied during embedding to limit the influence of any single document. "
        "Robust aggregation techniques like trimmed mean can reduce the impact of outlier retrievals.",
        NM))
    story.append(SP())

    story.append(Paragraph("5. Evaluation Framework for Robustness", H2))
    data = [["Attack Type", "Metric Affected", "Expected Drop", "Detection Method"],
            ["Label Flipping", "Faithfulness", "30-60%", "Semantic consistency"],
            ["Backdoor", "Answer Correctness", "50-90%", "Trigger word scanning"],
            ["Noise Injection", "Context Precision", "20-40%", "Embedding distance"],
            ["Semantic Poison", "All metrics", "10-30%", "Perplexity filtering"]]
    t = Table(data, colWidths=[1.6*inch, 1.6*inch, 1.4*inch, 1.8*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#B71C1C")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#FFEBEE"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(SP())

    story.append(Paragraph("6. Ethical Considerations", H2))
    story.append(Paragraph(
        "Research into data poisoning is conducted to improve AI system security, not to enable attacks. "
        "All poisoning experiments should be performed in isolated, controlled environments. "
        "Findings should be disclosed responsibly to relevant stakeholders. "
        "Regulatory frameworks such as the EU AI Act classify high-risk AI systems that require "
        "mandatory robustness testing before deployment.",
        NM))

    doc.build(story)
    print(f"Created: {path}")


if __name__ == "__main__":
    make_pdf1()
    make_pdf2()
    make_pdf3()
    print("\nAll 3 PDFs generated successfully in data/raw_pdfs/")