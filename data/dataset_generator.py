"""
Synthetic dataset generator for text embedding benchmarks.
Author: chandu
"""

import json
import os
import random
from typing import Dict, List


# Document templates for different topics
DOCUMENT_TEMPLATES = {
    "technology": [
        "Artificial intelligence and machine learning have revolutionized {}. Deep learning algorithms can now {} with unprecedented accuracy. Neural networks are being applied to {}. Researchers are developing new architectures that improve {}. The impact on {} has been transformative.",
        "Cloud computing platforms enable {} to scale efficiently. Distributed systems handle {} across multiple servers. Microservices architecture allows {} to be deployed independently. Container orchestration with Kubernetes manages {}. DevOps practices ensure {}.",
        "Cybersecurity threats continue to evolve with {}. Encryption protocols protect {} from unauthorized access. Multi-factor authentication prevents {}. Security teams monitor {} for suspicious activity. Zero-trust architecture assumes {}.",
        "Blockchain technology provides {} with decentralized trust. Smart contracts automate {} without intermediaries. Cryptocurrency transactions verify {} through consensus. Distributed ledgers record {} immutably. Applications extend beyond {} to supply chain management.",
        "The Internet of Things connects {} devices to networks. Sensors collect {} from the physical world. Edge computing processes {} locally. Real-time analytics enable {} decision-making. Industrial IoT optimizes {}.",
    ],
    "science": [
        "Quantum mechanics describes {} at the atomic scale. Wave-particle duality explains {}. The uncertainty principle limits {}. Quantum entanglement enables {}. Applications include quantum computing and {}.",
        "Climate change affects {} through rising temperatures. Greenhouse gas emissions contribute to {}. Renewable energy sources reduce {}. Carbon capture technologies mitigate {}. Global cooperation is needed to address {}.",
        "Genetic engineering allows {} to be modified precisely. CRISPR technology enables {}. DNA sequencing reveals {}. Gene therapy treats {} at the molecular level. Ethical considerations include {}.",
        "Astronomy explores {} beyond Earth. Telescopes observe {} across the electromagnetic spectrum. Exoplanets orbit {} in distant solar systems. Black holes warp {} with extreme gravity. The search for {} continues.",
        "Neuroscience investigates {} in the brain. Neurons communicate through {}. Synaptic plasticity enables {}. Brain imaging reveals {} activity patterns. Understanding {} can lead to new treatments.",
    ],
    "history": [
        "The Renaissance period brought {} to Europe. Artists like {} created masterpieces. Humanist philosophy emphasized {}. Scientific inquiry challenged {}. The printing press spread {}.",
        "World War II shaped {} in the 20th century. The conflict involved {} across continents. Technological advances included {}. The aftermath led to {}. International organizations were established to prevent {}.",
        "The Industrial Revolution transformed {} through mechanization. Steam engines powered {}. Factories centralized {}. Urbanization accelerated as {}. Social changes included {}.",
        "Ancient civilizations developed {} independently. The Egyptians built {} along the Nile. Mesopotamian societies invented {}. Chinese dynasties advanced {}. Roman engineering created {}.",
        "The Cold War divided {} between superpowers. Nuclear weapons created {}. Proxy conflicts occurred in {}. The space race drove {}. The collapse of {} ended the era.",
    ],
    "sports": [
        "Professional basketball requires {} skills. Players practice {} daily. Team strategies involve {}. Championships are won through {}. Athletic performance depends on {}.",
        "Soccer is played {} worldwide. The World Cup attracts {}. Tactics emphasize {}. Individual talent and {} both matter. Training regimens include {}.",
        "Olympic athletes train {} for years. Competition rules ensure {}. Records are broken through {}. Medal ceremonies honor {}. The Olympic spirit promotes {}.",
        "Tennis matches test {} between players. Serves can reach {}. Rally exchanges require {}. Grand Slam tournaments feature {}. Mental toughness determines {}.",
        "Marathon running demands {} endurance. Pacing strategies affect {}. Hydration is crucial for {}. Elite runners complete {} in under hours. Training builds {}.",
    ],
    "arts": [
        "Impressionist painters captured {} through brushstrokes. Light and color conveyed {}. Monet's work depicted {}. The movement rejected {}. Exhibitions showcased {}.",
        "Classical music composition requires {}. Symphonies develop {} over movements. Orchestras perform {} with precision. Conductors interpret {}. Baroque and {} periods have distinct styles.",
        "Film directors use {} to tell stories. Cinematography creates {} visually. Editing shapes {} narrative. Acting brings {} to life. Genre conventions include {}.",
        "Modern architecture emphasizes {} in design. Form follows {}. Sustainable buildings incorporate {}. Urban planning considers {}. Iconic structures become {}.",
        "Literary fiction explores {} through narrative. Character development reveals {}. Symbolism adds {}. Plot structure provides {}. Themes examine {}.",
    ],
    "business": [
        "Marketing strategies target {} audiences. Digital campaigns leverage {}. Brand awareness builds through {}. Customer analytics inform {}. ROI measures {}.",
        "Supply chain management optimizes {}. Logistics coordinate {}. Inventory control minimizes {}. Just-in-time systems reduce {}. Global trade requires {}.",
        "Financial planning ensures {} stability. Investment portfolios diversify {}. Risk management protects against {}. Cash flow monitoring tracks {}. Budgets allocate {}.",
        "Human resources develop {} workforce. Recruitment identifies {}. Training programs enhance {}. Performance reviews assess {}. Company culture influences {}.",
        "Entrepreneurship involves {} new ventures. Business plans outline {}. Funding sources include {}. Market research validates {}. Scaling challenges require {}.",
    ],
    "health": [
        "Cardiovascular health depends on {}. Exercise strengthens {}. Diet affects {}. Regular checkups detect {}. Preventive care reduces {}.",
        "Mental health treatment includes {}. Therapy helps process {}. Medications can manage {}. Stress reduction techniques involve {}. Support systems provide {}.",
        "Nutrition science studies {} effects on health. Vitamins support {}. Balanced diets include {}. Processed foods may {}. Hydration maintains {}.",
        "Public health initiatives prevent {}. Vaccination programs protect against {}. Sanitation improves {}. Health education promotes {}. Epidemiology tracks {}.",
        "Medical technology advances {} diagnosis. Imaging scans reveal {}. Minimally invasive procedures reduce {}. Telemedicine enables {} remotely. AI assists with {}.",
    ],
    "environment": [
        "Conservation efforts protect {} habitats. Endangered species require {}. Protected areas preserve {}. Restoration projects revive {}. Biodiversity supports {}.",
        "Pollution affects {} ecosystems. Air quality impacts {}. Water contamination threatens {}. Waste management reduces {}. Regulations limit {}.",
        "Sustainable agriculture produces {} responsibly. Organic farming avoids {}. Crop rotation maintains {}. Precision agriculture optimizes {}. Local food reduces {}.",
        "Deforestation destroys {} at alarming rates. Rainforests provide {}. Reforestation efforts plant {}. Indigenous communities depend on {}. Policy changes can protect {}.",
        "Ocean conservation addresses {} marine environments. Coral reefs support {}. Overfishing depletes {}. Plastic pollution harms {}. Marine reserves protect {}.",
    ],
    "education": [
        "Online learning platforms enable {} access to education. Video lectures explain {}. Interactive exercises reinforce {}. Certificates validate {}. Lifelong learning supports {}.",
        "STEM education prepares {} for careers. Hands-on projects teach {}. Critical thinking develops through {}. Collaboration skills build {}. Innovation requires {}.",
        "Early childhood education establishes {}. Play-based learning develops {}. Social skills emerge through {}. Reading readiness includes {}. Parental involvement supports {}.",
        "Higher education offers {} degree programs. Research universities advance {}. Liberal arts cultivate {}. Professional schools train {}. Student debt challenges {}.",
        "Educational technology transforms {} in classrooms. Tablets provide {}. Learning management systems organize {}. Adaptive software personalizes {}. Data analytics track {}.",
    ],
    "entertainment": [
        "Streaming services deliver {} on demand. Original content attracts {}. Binge-watching changes {}. Recommendation algorithms suggest {}. Subscription models fund {}.",
        "Video games offer {} interactive experiences. Graphics engines render {}. Multiplayer modes connect {}. Esports competitions feature {}. Game design balances {}.",
        "Live concerts create {} experiences. Sound systems deliver {}. Lighting design enhances {}. Venues accommodate {}. Tours bring {} to audiences.",
        "Podcasts cover {} diverse topics. Audio storytelling creates {}. Independent creators produce {}. Listeners subscribe to {}. Advertising supports {}.",
        "Theater productions present {} live performances. Actors rehearse {}. Stage design creates {}. Directors interpret {}. Broadway shows attract {}.",
    ],
}

# Fill-in options for templates
FILL_OPTIONS = [
    ["various industries", "data processing", "complex problems", "performance metrics", "society"],
    ["organizations", "massive workloads", "services", "containerized applications", "continuous delivery"],
    ["sophisticated attacks", "sensitive data", "unauthorized access", "network traffic", "no internal entity is trusted"],
    ["transactions", "agreements", "transactions", "all transactions", "finance"],
    ["billions of", "data", "data", "real-time", "manufacturing processes"],
    ["phenomena", "wave phenomena", "simultaneous measurements", "secure communication", "quantum cryptography"],
    ["ecosystems", "warming", "carbon footprint", "emissions", "climate challenges"],
    ["organisms", "precise edits", "genetic information", "genetic diseases", "designer babies"],
    ["celestial objects", "galaxies", "distant stars", "spacetime", "extraterrestrial life"],
    ["processes", "chemical signals", "learning", "neural", "consciousness"],
]


def generate_documents(num_docs: int, topics: List[str], seed: int = 42) -> List[Dict]:
    """Generate synthetic documents across topics."""
    random.seed(seed)
    documents = []

    docs_per_topic = num_docs // len(topics)

    doc_id = 1
    for topic in topics:
        templates = DOCUMENT_TEMPLATES.get(topic, DOCUMENT_TEMPLATES["technology"])

        for i in range(docs_per_topic):
            # Select random template
            template = random.choice(templates)

            # Fill in template with random options
            fill_values = random.choice(FILL_OPTIONS)
            try:
                text = template.format(*fill_values)
            except:
                # If formatting fails, use template as-is
                text = template

            # Add some topic-specific intro
            intros = [
                f"This article discusses {topic} in detail.",
                f"An overview of {topic} topics.",
                f"Key concepts in {topic}.",
                f"Understanding {topic} fundamentals.",
                f"Exploring {topic} in depth.",
            ]
            intro = random.choice(intros)

            full_text = f"{intro} {text}"

            documents.append({
                "id": f"doc_{doc_id:03d}",
                "text": full_text,
                "topic": topic
            })
            doc_id += 1

    return documents


def generate_queries(documents: List[Dict], num_queries: int, seed: int = 42) -> List[Dict]:
    """Generate queries with relevance judgments."""
    random.seed(seed)
    queries = []

    # Group documents by topic
    docs_by_topic = {}
    for doc in documents:
        topic = doc['topic']
        if topic not in docs_by_topic:
            docs_by_topic[topic] = []
        docs_by_topic[topic].append(doc)

    queries_per_topic = num_queries // len(docs_by_topic)

    query_templates = [
        "What is {} in {}?",
        "How does {} work in {}?",
        "Explain {} related to {}",
        "Tell me about {} in {}",
        "What are the key concepts of {} in {}?",
        "Describe {} in the context of {}",
        "How is {} used in {}?",
        "What are the applications of {} in {}?",
        "Discuss {} as it relates to {}",
        "Can you explain {} for {}?",
    ]

    query_id = 1
    for topic, topic_docs in docs_by_topic.items():
        for i in range(queries_per_topic):
            # Create query about this topic
            template = random.choice(query_templates)
            concept = random.choice(["systems", "methods", "techniques", "approaches", "principles",
                                   "technologies", "processes", "solutions", "strategies", "frameworks"])
            query_text = template.format(concept, topic)

            # Select relevant documents (1-3 from same topic)
            num_relevant = random.randint(1, 3)
            relevant_docs = random.sample(topic_docs, min(num_relevant, len(topic_docs)))

            # Create relevance judgments
            relevance = {}
            for doc in relevant_docs:
                # Assign relevance scores (3=highly, 2=relevant, 1=marginally)
                score = random.choice([2, 3])  # Higher probability of being relevant
                relevance[doc['id']] = score

            # Occasionally add marginally relevant docs from other topics
            if random.random() < 0.3:  # 30% chance
                other_topics = [t for t in docs_by_topic.keys() if t != topic]
                if other_topics:
                    other_topic = random.choice(other_topics)
                    marginal_doc = random.choice(docs_by_topic[other_topic])
                    relevance[marginal_doc['id']] = 1

            queries.append({
                "id": f"q_{query_id:03d}",
                "text": query_text,
                "topic": topic,
                "relevance": relevance
            })
            query_id += 1

    return queries


def generate_dataset(config: dict) -> dict:
    """
    Generate complete synthetic dataset.

    Args:
        config: Configuration dictionary with dataset parameters

    Returns:
        Dataset dictionary with documents and queries
    """
    dataset_config = config.get('dataset', {})
    num_docs = dataset_config.get('num_documents', 500)
    num_queries = dataset_config.get('num_queries', 100)
    seed = dataset_config.get('seed', 42)
    topics = dataset_config.get('topics', [
        'technology', 'science', 'history', 'sports', 'arts',
        'business', 'health', 'environment', 'education', 'entertainment'
    ])

    print(f"Generating {num_docs} documents across {len(topics)} topics...")
    documents = generate_documents(num_docs, topics, seed)

    print(f"Generating {num_queries} queries with relevance judgments...")
    queries = generate_queries(documents, num_queries, seed)

    dataset = {
        "documents": documents,
        "queries": queries,
        "metadata": {
            "num_documents": len(documents),
            "num_queries": len(queries),
            "topics": topics,
            "seed": seed
        }
    }

    return dataset


def load_or_generate_dataset(config: dict, cache_path: str = "data/datasets/synthetic_corpus.json") -> dict:
    """
    Load dataset from cache or generate new one.

    Args:
        config: Configuration dictionary
        cache_path: Path to cache file

    Returns:
        Dataset dictionary
    """
    # Check if cached dataset exists
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, 'r') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset['documents'])} documents and {len(dataset['queries'])} queries")
        return dataset

    # Generate new dataset
    print("No cached dataset found. Generating new dataset...")
    dataset = generate_dataset(config)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {cache_path}")

    return dataset


if __name__ == "__main__":
    # Test dataset generation
    test_config = {
        'dataset': {
            'num_documents': 50,
            'num_queries': 10,
            'seed': 42,
            'topics': ['technology', 'science', 'history']
        }
    }

    dataset = generate_dataset(test_config)
    print(f"\nGenerated {len(dataset['documents'])} documents")
    print(f"Generated {len(dataset['queries'])} queries")
    print(f"\nSample document: {dataset['documents'][0]}")
    print(f"\nSample query: {dataset['queries'][0]}")
