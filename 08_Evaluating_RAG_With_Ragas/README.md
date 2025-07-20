
<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 8: Evaluating RAG with Ragas</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 8: Pre-Work](https://www.notion.so/Session-8-RAG-Evaluation-and-Assessment-21dcd547af3d80b39f09ffa61d4bab89?source=copy_link#232cd547af3d80098545d9f576bc622f)| [Session 8: RAG Evaluation and Assessment](https://www.notion.so/Session-8-RAG-Evaluation-and-Assessment-21dcd547af3d80b39f09ffa61d4bab89) | [Recording!](https://us02web.zoom.us/rec/share/sNualc0Z-OE6TOWZvjiJoX71PVAMtMxiEcLfgFQhIpvbe634xYBwfwu-BEf-1tsO.7od_74gguSzFMCcL) (pDcv@!2j) | [Session 8 Slides](https://www.canva.com/design/DAGtcxIQkZs/giWGUaQrsRf5Aq2IoQV46Q/edit?utm_content=DAGtcxIQkZs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 8: RAG Evaluation and Assessment](https://forms.gle/RPWURmmrwg85ngdT6) | [AIE7 Feedback 7/17](https://forms.gle/oG2iR4Bkf9YpvA7R9) |

In today's assignment, we'll be creating Synthetic Data, and using it to benchmark (and improve) a LCEL RAG Chain.

- 🤝 Breakout Room #1
  1. Task 1: Installing Required Libraries
  2. Task 2: Set Environment Variables
  3. Task 3: Synthetic Dataset Generation for Evaluation using Ragas
  4. Task 4: Evaluating our Pipeline with Ragas
  5. Task 6: Making Adjustments and Re-Evaluating

- 🤝 Breakout Room #2
  1. Task 1: Building a ReAct Agent with Metal Price Tool
  2. Task 2: Implementing the Agent Graph Structure
  3. Task 3: Converting Agent Messages to Ragas Format
  4. Task 4: Evaluating Agent Performance using Ragas Metrics
     - Tool Call Accuracy
     - Agent Goal Accuracy  
     - Topic Adherence

## Ship 🚢

The completed notebook!

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

> NOTE: Completing this challenge will provide full marks on the assignment, regardless of the completion of the notebook. You do not need to complete this in the notebook for full marks.

##### **MINIMUM REQUIREMENTS**:

1. Baseline `LangGraph RAG` Application using `NAIVE RETRIEVAL`
2. Baseline Evaluation using `RAGAS METRICS`
  - [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)
  - [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html)
  - [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html)
  - [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html)
  - [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html)
3. Implement a `SEMANTIC CHUNKING STRATEGY`.
4. Create an `LangGraph RAG` Application using `SEMANTIC CHUNKING` with `NAIVE RETRIEVAL`.
5. Compare and contrast results.

##### **SEMANTIC CHUNKING REQUIREMENTS**:

Chunk semantically similar (based on designed threshold) sentences, and then paragraphs, greedily, up to a maximum chunk size. Minimum chunk size is a single sentence.

Have fun!
</details>

### Deliverables

- A short Loom of either:
  - the two notebooks for the main homework assignment; or
  - the notebook you created for the Bonus Challenge

## Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped Synthetic Data Generation, benchmarking, and iteration with RAGAS & LangChain! 🎉🤖

🔍 Three Key Takeaways:
1️⃣ 
2️⃣ 
3️⃣ 

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

## Submitting Your Homework

### Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s08-assignment`
2. Respond to the questions in the `Evaluating_RAG_with_Ragas_(2025)_AI_Makerspace.ipynb` notebook:
    + Edit the markdown cells with the questions then enter your responses
    + NOTE: Remember to create a header (example: `##### ✅ Answer:`) to help the grader find your responses
3. Respond to the questions in the `Evaluating_Agents_with_Ragas_(2025)_AI_Makerspace.ipynb` notebook:
    + Edit the markdown cells with the questions then enter your responses
    + NOTE: Remember to create a header (example: `##### ✅ Answer:`) to help the grader find your responses
4. Commit, and push your completed notebook to your `origin` repository. _NOTE: Do not merge it into your main branch._
5. Record a Loom video reviewing the content of your completed notebooks
6. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the `08_Evaluating_RAG_With_Ragas` folder _on your assignment branch (not main)_
    + The URL to your Loom Video
    + Your Three lessons learned/not yet learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_

### Advanced Build

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s08-assignment`
2. Create a notebook that meets or exceeds the **MINIMUM REQUIREMENTS** as well as the **SEMANTIC CHUNKING REQUIREMENTS**
3. Commit, and push your completed notebook to your `origin` repository. _NOTE: Do not merge it into your main branch._
4. Record a Loom video reviewing the content of your completed notebook.
5. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the notebook you created for the Bonus Challenge _on your assignment branch_
    + The URL to your Loom Video
    + Your Three lessons learned/not yet learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_
