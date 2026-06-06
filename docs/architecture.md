# SkyMetrics — Architecture

All diagrams are derived from the actual modules under `src/skymetrics/`.

## Component / architecture diagram

```mermaid
flowchart TB
    subgraph Sources["Data sources"]
        SKY["Skytrax reviews"]
        OS["OpenSky Network"]
        IAG["IAG share price (Yahoo)"]
        CSV["Forage booking CSV (50k)"]
    end

    subgraph Ingest["Ingestion — scrapers/"]
        SCRAPE["skytrax.py\nfetch + parse"]
        EXT["external.py\nOpenSky / IAG"]
    end

    subgraph Core["Processing"]
        NLP["nlp/\nsentiment · aspects · topics · transformer"]
        ML["ml/\nbooking · predict · importance · model_card · persistence"]
        LLM["llm/\nsummarize · rag (TF-IDF retrieval)"]
    end

    OR["OpenRouter\n(free models)"]

    DB[("db/\nSQLite: reviews · bookings · metrics")]

    subgraph Serve["Serving"]
        API["api/app.py\nFastAPI"]
        DASH["dashboard/app.py\nStreamlit"]
    end

    SKY --> SCRAPE --> DB
    OS --> EXT --> DB
    IAG --> EXT
    CSV --> ML
    DB --> NLP
    DB --> ML
    NLP --> API
    NLP --> DASH
    ML --> API
    DB --> LLM
    LLM --> OR
    LLM --> API
    LLM --> DASH
    OS --> API
    DB --> DASH
```

## Data flow diagram (DFD)

```mermaid
flowchart LR
    R[/"Raw reviews HTML"/] -->|parse_reviews| C["Clean text"]
    C -->|polarity_score| S["Sentiment label + score"]
    C -->|aspect_sentiment| A["Per-aspect scores"]
    C -->|top_terms| T["TF-IDF topics"]
    B[/"Booking CSV"/] -->|prepare_features| F["Encoded features"]
    F -->|train_pipeline| M["RF model + metrics"]
    S --> DB[("SQLite")]
    A --> DB
    M -->|save_model| ART["model.joblib"]
    DB --> UI["Dashboard / API responses"]
    ART --> API["/predict (future)"]
```

## Core sequence — review analysis request

```mermaid
sequenceDiagram
    participant U as Client
    participant API as FastAPI app
    participant N as nlp.sentiment / nlp.aspects
    U->>API: POST /sentiment {text}
    API->>N: polarity_score(text)
    N-->>API: polarity
    API->>N: classify_sentiment(polarity)
    N-->>API: label
    API-->>U: {polarity, label}
    U->>API: POST /aspects {text}
    API->>N: aspect_sentiment(text)
    N-->>API: {aspect: score, ...}
    API-->>U: {aspects}
```

## ER diagram — SQLite schema

```mermaid
erDiagram
    REVIEWS {
        int id PK
        text source
        text review
        real rating
        real polarity
        text label
        text review_date
    }
    BOOKINGS {
        int id PK
        text route
        text booking_origin
        int purchase_lead
        int length_of_stay
        real flight_duration
        int booking_complete
    }
    METRICS {
        int id PK
        text name
        real value
        text period
        text source
    }
```
