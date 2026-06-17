# DualEncoderSingleDecoder
## <ins> Context Aware / task specific object detection framework, based on Dual Encoder Single Decoder architecture. </ins>

This repository contains the training code for the task specific object detection framework whose Dual Encoder Single Decoder architecture is heavily inspired by CLIP and DETR papers.


# Architecture

The architecture is a lightweight dual-encoder system designed to preserve semantic grounding capabilities while maintaining a low parameter count. It consists of three main pathways:

1. **Text Pathway (Task Encoding):** Converts natural language prompts into semantic token embeddings using a frozen text encoder (e.g., DistilBERT).
2. **Vision Pathway (Visual Feature Extraction):** Extracts spatial feature maps from input images using a lightweight CNN backbone.
3. **Transformer Grounding Head:** A cross-attention module that fuses the text queries with visual keys/values to predict the region of interest.

### Architecture Diagram

```mermaid
flowchart TD
    %% Define Styles
    classDef textPath fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef visionPath fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef fusionPath fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef outPath fill:#fff3e0,stroke:#f57c00,stroke-width:2px;

    %% Text Pipeline
    subgraph Text Pipeline
        A["Task Prompt"] --> B["Text Encoder<br><i>DistilBERT / CLIP-Text</i>"]
        B --> C["Task Embedding<br><b>Query (Q)</b>"]
    end
    class A,B,C textPath;

    %% Vision Pipeline
    subgraph Vision Pipeline
        D["Input Image"] --> E["CNN Backbone<br><i>MobileNetV3 / EfficientNet-lite</i>"]
        E --> F["Dense Visual Features<br><b>H x W x C</b>"]
        F --> G["Feature Flattening<br><i>Spatial to Token Sequence</i>"]
        G --> H["Image Feature Tokens<br><b>Keys (K) & Values (V)</b>"]
    end
    class D,E,F,G,H visionPath;

    %% Grounding Decoder
    subgraph Transformer Grounding Decoder
        C -->|Query| I["Transformer Attention Block<br><i>2-4 Layers Cross-Attention</i>"]
        H -->|Keys & Values| I
    end
    class I fusionPath;

    %% Prediction Head
    subgraph Prediction
        I --> J["Bounding Box Head<br><i>Fully Connected</i>"]
        J --> K["Final Output<br><b>Bounding Box (x, y, w, h)</b>"]
    end
    class J,K outPath;
```

# Some Results:
![Some results alt txt](SomeResults.png)
_ToDo: Add more description about the project, training methods, evalution, curves etc_
