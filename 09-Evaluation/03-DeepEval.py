from deepeval.metrics import ContextualPrecisionMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Define test case
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day no-questions-asked full refund service.",
    expected_output="Customers can return the goods within 30 days and get a full refund.",
    retrieval_context=["All customers are eligible for a 30-day no-questions-asked full refund service."]
)

# Define evaluation metrics
contextual_precision = ContextualPrecisionMetric()
answer_relevancy = AnswerRelevancyMetric()

# Run evaluation
contextual_precision.measure(test_case)
answer_relevancy.measure(test_case)

print("Contextual Precision Score: ", contextual_precision.score)
print("Answer Relevancy Score: ", answer_relevancy.score)