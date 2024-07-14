# Mesop Langgraph Chatbot
## Goal
- Implement MoA with langgraph
- Demonstrate MoA Chatbot with Mesop

## Installation
1. Install [poetry](https://python-poetry.org/docs/#installation)
2. Run `poetry install` in this directory.
3. Run `poetry shell` to use venv

## Run demo
1. `mesop src/mesop_langgraph/main.py`
2. Chat!


## Design
### MoA Architecture
```mermaid
flowchart LR
    question --> proposers1 & proposer2 & proposer3 --> aggregator --> answer
```
- Proposer Layer: 1
- Aggregator Layer: 1
### LLMs
#### Proposers
- [qwen2:7b-instruct](https://ollama.com/library/qwen2:7b-instruct)
    - temperature: 0.2
- [mistral:7b-instruct](https://ollama.com/library/mistral:instruct)
    - temperature: 0.2
- [llama3:8b-instruct](https://ollama.com/library/llama3:instruct)
    - temperature: 0.2
#### Aggregator
- [qwen2:7b-instruct](https://ollama.com/library/qwen2:7b-instruct)
    - temperature: 0.2
### Prompts
Aggregate-and-Synthesize Prompt
```plaintext
<<SYSTEM>>
You have been provided with a set of responses from various open-source models to the latest user query. Your
task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the
information provided in these responses, recognizing that some of it may be biased or incorrect. Your response
should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply
to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of
accuracy and reliability.
Responses from models:
1. [Model Response from Ai,1]
2. [Model Response from Ai,2]
...
n. [Model Response from Ai,n]

Human: [x_i]
```

## References
- [Langgraph](https://langchain-ai.github.io/langgraph/)
- [Mesop](https://google.github.io/mesop/)