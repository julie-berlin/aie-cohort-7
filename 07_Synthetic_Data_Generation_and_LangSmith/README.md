
<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 7: Synthetic Data Generation and LangSmith</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 7: Pre-Work](https://www.notion.so/Session-7-Synthetic-Data-Generation-for-Evaluation-21dcd547af3d802bb7f8f0e78a27c305?source=copy_link#22dcd547af3d801b8dc3f5ba42a1d9ef)| [Session 7: Synthetic Data Generation for Evaluation](https://www.notion.so/Session-7-Synthetic-Data-Generation-for-Evaluation-21dcd547af3d802bb7f8f0e78a27c305) | Coming soon! | Coming soon! | You are here! | Coming soon! | Coming soon!

In today's assignment, we'll be creating Synthetic Data, and using it to benchmark (and improve) a LCEL RAG Chain.

- 🤝 BREAKOUT ROOM #1
  1. Use RAGAS to Generate Synthetic Data

- 🤝 BREAKOUT ROOM #2
  1. Load them into a LangSmith Dataset
  2. Evaluate our RAG chain against the synthetic test data
  3. Make changes to our pipeline
  4. Evaluate the modified pipeline

## Ship 🚢

The completed notebook!

### 🚧 OPTIONAL: Advanced Build

Reproduce the RAGAS Synthetic Data Generation Steps - but utilize a LangGraph Agent Graph, instead of the Knowledge Graph approach.

This generation should leverage the [Evol Instruct](https://arxiv.org/pdf/2304.12244) method to generate synthetic data.

Your final state (output) should contain (at least, not limited to):

1. `List(dict)`: Evolved Questions, their IDs, and their Evolution Type.
2. `List(dict)`: Question IDs, and Answer to the referenced Evolved Question.
3. `List(dict)`: Question IDs, and the relevant Context(s) to the Evolved Question.

The Graph should handle:

1. Simple Evolution.
2. Multi-Context Evolution.
3. Reasoning Evolution.

It should take, as input, a list of LangChain Documents.

### Deliverables

- A short Loom of the notebook

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
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s07-assignment`
2. Respond to the activities and questions in the `Synthetic_Data_Generation_RAGAS_&_LangSmith_Assignment.ipynb` notebook:
    + Edit the markdown cells of the activities and questions then enter your responses
    + Edit/Create code cell(s) where necessary as part of an activity
    + NOTE: Remember to create a header (example: `##### ✅ Answer:`) to help the grader find your responses
3. Commit, and push your completed notebook to your `origin` repository. _NOTE: Do not merge it into your main branch._
4. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the completed notebook _on your assignment branch (not main)_
    + The URL to your Loom Video
    + Your Three lessons learned/not yet learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_

### Advanced Build
1. Include a 1 minute walkthrough of your completed application as part of your Main Homework Assignment's Loom video
2. In addition to the Homework Submission instructions in Main Homework Assignment ➡ Step 4, include the following URLs to your Advanced Build's:
    + GitHub Repo
    + Production Deployment
