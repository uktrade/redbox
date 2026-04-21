This Mermaid diagram doesn't render in GitHub because it uses the 
[elk layout](https://mermaid.js.org/intro/syntax-reference.html#supported-layout-algorithms) which is more sophisticated but not supported by
[GitHub](https://github.com/orgs/community/discussions/138426).

If you want to make a change to this diagram, you can paste to https://mermaid.live/ 
or use the VSCode [Mermaid extension](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid)

```mermaid
---
config:
  layout: elk
  theme: redux
  look: neo
  themeCSS: |
    /* Thinner subgraph borders */
    .cluster rect {
      stroke-width: 0.3px !important;
    }

    /* Bigger subgraph titles */
    .cluster, [class*="cluster"] text {
      font-size: 18px;
    }
---
flowchart LR
    Browser["🌐 <br>Browser"]

    subgraph DjangoApp["Django App"]
        UPLOAD["/upload/"]
        CHAT["/ws/chat/"]
    end

    subgraph PostgreSQL["PostgreSQL"]
        FileRecord[("File <br>Records")]
        DjangoQ[("Task <br>Queue")]
    end

    S3@{ shape: trap-t, label: "AWS S3" }

    subgraph Worker["ETL Worker"]
        Extract
        Chunk
        Embed
    end

    OpenSearch[("OpenSearch")]
    Textract["AWS Textract"]
    Bedrock["AWS <br>Bedrock"]

    Browser --> UPLOAD
    Browser --> CHAT       
    CHAT -- "lookup <br>file keys" --> FileRecord                               
    CHAT -- "query by file keys" --> OpenSearch

    UPLOAD --file--> S3
    UPLOAD --metadata--> FileRecord
    FileRecord --queue <br>task--> DjangoQ
    DjangoQ --file ID--> Extract


    Worker -- "Update status" --> FileRecord
    Embed --> OpenSearch
    Embed <--> Bedrock
    Textract --> Extract
    S3 --PDF--> Textract
    S3 --non-PDF--> Extract
    Extract --> Chunk
    Chunk --> Embed
```
