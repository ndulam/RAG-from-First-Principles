import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI as OpenAIClient  # Avoid naming conflicts with TruLens's OpenAI class

# Trulens is an open-source library for observability and evaluation of deep learning models (especially LLM applications).
# It helps developers track, debug, and evaluate the performance of complex applications like RAG (Retrieval Augmented Generation).
# - TruSession: Manages evaluation sessions and result storage.
# - Feedback: Defines evaluation metrics such as relevance, groundedness, etc.
# - TruApp: Wraps your application to make it monitorable by Trulens.
# - instrument: A decorator used to mark specific functions that need to be tracked.
from trulens.core import TruSession, Feedback, Select
from trulens.apps.app import TruApp, instrument
from trulens.providers.openai import OpenAI as TruLensOpenAI
import numpy as np

# Set API key
# os.environ["OPENAI_API_KEY"] = "your_key_here"

# Initialize embedding function
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get("OPENAI_API_KEY"),
                                             model_name="text-embedding-ada-002")
chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection("Info", embedding_function=embedding_function)

# Add example data
vector_store.add("starbucks_info", documents=[
    """
    Starbucks Corporation is an American multinational chain of coffeehouses headquartered in Seattle, Washington.
    As the world's largest coffeehouse chain, Starbucks is seen to be the main representation of the United States' second wave of coffee culture.
    """
])

class RAG:
    # @instrument is a decorator provided by Trulens to "instrument" or "equip" a function.
    # When marked, every time this function is called, Trulens records its input, output, execution time, errors, and other information.
    # This is crucial for understanding the specific behavior of each step (retrieval, generation) in the RAG process.
    @instrument
    def retrieve(self, query: str):
        """Retrieve relevant documents"""
        results = vector_store.query(query_texts=[query], n_results=2)
        return results["documents"][0] if results["documents"] else []

    @instrument
    def generate_completion(self, query: str, context: list):
        """Generate answer"""
        oai_client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))
        context_str = "\n".join(context) if context else "No context available."
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Context: {context_str}\nQuestion: {query}"}]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str):
        """Complete RAG query process"""
        context = self.retrieve(query)
        return self.generate_completion(query, context)

# Initialize TruLens session
# TruSession is the entry point for interacting with the Trulens backend (which can be a local SQLite file or a database).
# It is responsible for managing and storing all tracking data and evaluation results.
# The database_redact_keys=True option automatically hides sensitive information (like API keys) in records to ensure security.
session = TruSession(database_redact_keys=True)
# Resetting the database clears all previous records, ensuring we start this evaluation from a clean environment.
session.reset_database()

# Initialize TruLens's OpenAI provider
# A Provider is Trulens's "judge" for performing evaluations. Here we use OpenAI's gpt-4 model.
# These evaluations themselves are done by asking the LLM questions, such as: "Is the given answer consistent with the context?"
provider = TruLensOpenAI(model_engine="gpt-4")

# Define evaluation metrics (Feedback Functions)
# Feedback is a core concept in Trulens, used to define the evaluation dimensions we care about.
# Each Feedback function consists of an evaluator (a method of the provider) and a selector.
# The selector (.on(...)) precisely specifies which part of the application (input, output, or intermediate results) to evaluate.

# 1. Groundedness
#    - Evaluator: provider.groundedness_measure_with_cot_reasons, uses CoT (Chain of Thought) to determine if the answer is fully based on the provided context.
#    - Selector: .on(Select.RecordCalls.retrieve.rets) specifies that the context for evaluation is the return result of the `retrieve` method.
#              .on_output() specifies that the content to be evaluated is the final output of the RAG application (the generated answer).
f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") \
    .on(Select.RecordCalls.retrieve.rets).on_output()

# 2. Answer Relevance
#    - Evaluator: provider.relevance_with_cot_reasons, determines if the generated answer is relevant to the original question.
#    - Selector: .on_input() specifies that the context for evaluation is the top-level input of the application (the user's query).
#              .on_output() specifies that the content to be evaluated is the final output of the application.
f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance") \
    .on_input().on_output()

# 3. Context Relevance
#    - Evaluator: provider.context_relevance_with_cot_reasons, determines the relevance of the retrieved context to the original question.
#    - Selector: .on_input() specifies that the context for evaluation is the top-level input of the application.
#              .on(Select.RecordCalls.retrieve.rets[:]) specifies that the content to be evaluated is each element in the list of contexts returned by the `retrieve` method.
#    - Aggregator: .aggregate(np.mean) Because the context may contain multiple documents, an aggregation function (here, calculating the mean) is used to combine all context relevance scores into a single total score.
f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") \
    .on_input().on(Select.RecordCalls.retrieve.rets[:]).aggregate(np.mean)

# Set up TruApp
# TruApp packages our RAG application instance with the list of Feedback functions defined above.
# It creates an "observable" application that can be tracked and evaluated by Trulens.
rag = RAG()
tru_rag = TruApp(
    rag,
    app_name="RAG",
    app_version="base",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)

# Execute query and record
# Use the `with tru_rag as recording:` context manager to run the application.
# Within this code block, calls to `rag.query()` and all its internally `@instrument` marked methods will be automatically recorded by Trulens.
# The recorded data (app-json) includes the complete call chain, inputs, outputs, and intermediate results.
with tru_rag as recording:
    response = rag.query("What wave of coffee culture is Starbucks seen to represent in the United States?")
    print(f"Response: {response}")

# View evaluation results
# The get_leaderboard() method reads records from the database and displays all evaluation results in a table format.
# It shows the average score for each Feedback, allowing us to quickly understand the overall performance of the application.
# This leaderboard is very useful for comparing performance differences between different versions of an application (e.g., after changing prompts, models, or retrieval strategies).
print(session.get_leaderboard())