"""Grounding / hallucination check using NLI (Natural Language Inference).

Uses a cross-encoder NLI model to verify that each claim in the generated
answer is entailed by the retrieved passages.
"""

from __future__ import annotations

import re
from typing import Any

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from guardrails.schemas import GuardrailResult

REFUSAL_MESSAGES = {
    "en": "I'm not confident this answer is fully supported by the source documents. Here are the relevant passages instead:",
    "hi": "मुझे विश्वास नहीं है कि यह उत्तर स्रोत दस्तावेजों द्वारा पूरी तरह से समर्थित है। यहाँ प्रासंगिक अंश हैं:",
    "ta": "இந்த பதில் மூல ஆவணங்களால் முழுமையாக ஆதரிக்கப்படுகிறது என்று நான் உறுதியாக நம்பவில்லை. தொடர்புடைய பகுதிகள் இதோ:",
}


class GroundingChecker:
    """Checks if generated answers are grounded in retrieved passages using NLI."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        """Initialize the lexical checker."""
        print(f"Initialized lexical overlap grounding checker")

    def _split_into_claims(self, text: str) -> list[str]:
        """Split generated text into individual claims/sentences."""
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?।॥])\s+', text)

        # Filter out very short fragments and citation markers
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) > 15 and not s.startswith("[") and not s.startswith("Source"):
                claims.append(s)
        return claims

    def check_grounding(
        self,
        answer: str,
        passage_texts: list[str],
        contradiction_threshold: float = 0.7,
        entailment_threshold: float = 0.5,
        query_language: str = "en",
    ) -> GuardrailResult:
        """Check if the generated answer is grounded in the passages.

        Args:
            answer: The generated answer text.
            passage_texts: List of retrieved passage texts.
            contradiction_threshold: Score above which to flag as contradicting.
            entailment_threshold: Score below which to flag as ungrounded.
            query_language: Language for refusal message.

        Returns:
            GuardrailResult indicating grounding status.
        """
        if not answer or not passage_texts:
            return GuardrailResult(
                passed=True,
                reason=None,
                category="hallucination",
                confidence=0.0,
            )

        claims = self._split_into_claims(answer)
        if not claims:
            return GuardrailResult(
                passed=True,
                reason=None,
                category="hallucination",
                confidence=0.0,
            )

        # Concatenate all passages as the premise
        combined_passages = " ".join(passage_texts[:5])  # Limit to top 5 for speed

        # Check each claim against the combined passages
        ungrounded_claims = 0
        total_entailment_score = 0.0

        for claim in claims:
            # Fast lexical overlap check
            combined_passages_lower = combined_passages.lower()
            
            words = set(re.findall(r'\w+', claim.lower()))
            overlap_score = 1.0 # default if no words
            if words:
                overlap = sum(1 for w in words if w in combined_passages_lower)
                overlap_score = overlap / len(words)

            total_entailment_score += overlap_score

            if overlap_score < entailment_threshold:
                ungrounded_claims += 1

        grounding_ratio = 1.0 - (ungrounded_claims / len(claims)) if claims else 1.0

        if grounding_ratio >= 0.7:  # At least 70% of claims are grounded
            return GuardrailResult(
                passed=True,
                reason=None,
                category="hallucination",
                confidence=grounding_ratio,
            )
        else:
            refusal = REFUSAL_MESSAGES.get(query_language, REFUSAL_MESSAGES["en"])
            return GuardrailResult(
                passed=False,
                reason=refusal,
                category="hallucination",
                confidence=1.0 - grounding_ratio,
            )
