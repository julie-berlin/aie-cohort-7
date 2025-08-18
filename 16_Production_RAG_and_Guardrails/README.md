<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 16: Production RAG and Guardrails</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE7/00_AIM_Quicklinks)

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 16: Pre-Work](https://www.notion.so/Session-16-Guardrails-and-Caching-24ecd547af3d8096a41ad50908e4680e?source=copy_link#24ecd547af3d8181a0a3cf1182096983)| [Session 16: Guardrails and Caching](https://www.notion.so/Session-16-Guardrails-and-Caching-24ecd547af3d8096a41ad50908e4680e) | [Recording!](https://us02web.zoom.us/rec/share/FvwwIxAOfnhHakLdTwVNrvga6lDk1hslp13SzVWxlseGA-LBv3mWTeXBzrQ5EUmn.DJUzMMZci0ZAocLl) (6dbE5%1W)| [Session 16 Slides](https://www.canva.com/design/DAGwFTcRpeI/8rXGxPxvkfTXosIPZFJNAQ/edit?utm_content=DAGwFTcRpeI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 16 Assignment: Production RAG and Guardrails](https://forms.gle/C5zfhdZJdVkrZMJG9) | [AIE7 Feedback 8/14](https://forms.gle/9LrV7pAxZ3vpU7B98)

# Build 🏗️

Run the notebook and complete the following:

1. 🤝 BREAKOUT ROOM #1:
  - Task 1: Dependencies and Set-Up
  - Task 2: Setting up Production RAG and LangGraph Agent Integration
  - Task 3: LangGraph Agent Integration
2. 🤝 BREAKOUT ROOM #2:
  - Task 4: Guardrails Integration for Production Safety

## Guard Rails Set-up

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Guardrails API

```bash
uv run guardrails configure
```

Provide your Guardrails AI API key, found [here](https://hub.guardrailsai.com/keys).

### 3. Install Required Guards

```bash
uv run guardrails hub install hub://tryolabs/restricttotopic
uv run guardrails hub install hub://guardrails/detect_jailbreak
uv run guardrails hub install hub://guardrails/competitor_check
uv run guardrails hub install hub://arize-ai/llm_rag_evaluator
uv run guardrails hub install hub://guardrails/profanity_free
uv run guardrails hub install hub://guardrails/guardrails_pii
```

## 🚧 Advanced Build:

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

The caching we're using is both: 

1. Ineffecient
2. Exact Match

Please produce a locally running application (through Docker) that integrates a more intelligent caching process.

In simpler terms: 

- Use a database approach (Redis, Vectordatase, SQLite, etc.) instead of plain-memory for caching
- Implement Semantic LLM Caching OR Implement E2E Caching

> NOTE: Doing the advanced build will count as your assignment for the week. If you do the advanced build, you are not required to do the notebook.

</details>

# Ship 🚢

- Graph including Guardrail nodes in repository
- 5min. Loom Video

# Share 🚀
- Walk through your notebook and explain what you've completed in the Loom video
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

# Submitting You Homework

## Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s16-assignment`
2. Respond to the questions in the `Prototyping_LangChain_Application_with_Production_Minded_Changes_Assignment.ipynb` notebook:
    + Edit the markdown cells with the questions then enter your responses
    + NOTE: Remember to create a header (example: `##### ✅ Answer:`) to help the grader find your responses
3. Complete the activities in the notebook
4. Commit, and push your completed notebook to your `origin` repository. _NOTE: Do not merge it into your main branch_
5. Record a Loom video reviewing:
    + the caching and guardrail concepts you have learned
    + the content of your completed notebook
6. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the `16_LLMOps` folder _on your assignment branch (not main)_
    + The URL to your Loom Video
    + Your Three lessons learned/not yet learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_

## OPTIONAL: Advanced Build Assignment _(Can be done in lieu of the Main Homework Assignnment)_

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s16-assignment`
2. Create a locally running application (through Docker) that integrates a more intelligent caching process
3. Commit, and push your application code to your `origin` repository. _NOTE: Do not merge it into your main branch_
4. Record a Loom video reviewing your application
5. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the `16_LLMOps` folder _on your assignment branch_
    + The URL to your Loom Video
    + Your Three lessons learned/not yet learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_
