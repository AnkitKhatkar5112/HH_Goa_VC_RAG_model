"""Off-topic query detection via retrieval similarity threshold.

Uses the same FAISS index — if the top-1 retrieval score is below
a calibrated threshold, the query is flagged as off-topic.
"""

from __future__ import annotations

from guardrails.schemas import GuardrailResult

# Refusal messages in supported languages
REFUSAL_MESSAGES = {
    "en": "I can only answer questions related to the topics in my knowledge base. Your question appears to be outside my scope.",
    "hi": "मैं केवल अपने ज्ञान आधार से संबंधित प्रश्नों का उत्तर दे सकता हूँ। आपका प्रश्न मेरे दायरे से बाहर प्रतीत होता है।",
    "ta": "என் அறிவுத் தளத்தில் உள்ள தலைப்புகள் தொடர்பான கேள்விகளுக்கு மட்டுமே பதிலளிக்க முடியும். உங்கள் கேள்வி என் எல்லைக்கு வெளியே இருப்பதாகத் தெரிகிறது.",
}


def check_off_topic(
    top_score: float,
    threshold: float = 0.35,
    query_language: str = "en",
) -> GuardrailResult:
    """Check if a query is off-topic based on its best retrieval score.

    Args:
        top_score: The highest similarity score from vector search.
        threshold: Minimum score to consider a query on-topic.
        query_language: Language for the refusal message.

    Returns:
        GuardrailResult indicating whether the query passed.
    """
    is_on_topic = top_score >= threshold

    if is_on_topic:
        return GuardrailResult(
            passed=True,
            reason=None,
            category="off_topic",
            confidence=top_score,
        )
    else:
        refusal = REFUSAL_MESSAGES.get(query_language, REFUSAL_MESSAGES["en"])
        return GuardrailResult(
            passed=False,
            reason=refusal,
            category="off_topic",
            confidence=1.0 - top_score,  # confidence that it IS off-topic
        )
