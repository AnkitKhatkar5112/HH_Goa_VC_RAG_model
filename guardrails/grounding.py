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
        """Initialize the NLI model.

        Args:
            model_name: HuggingFace cross-encoder model for NLI.
        """
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.label_mapping = ["contradiction", "entailment", "neutral"]
        print(f"Loaded NLI model: {model_name}")

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
            pair = [(combined_passages, claim)]
            scores = self.model.predict(pair)

            # scores is [contradiction_score, entailment_score, neutral_score]
            if hasattr(scores[0], '__len__'):
                contradiction_score = float(scores[0][0])
                entailment_score = float(scores[0][1])
            else:
                # Single score model — assume higher = more entailed
                entailment_score = float(scores[0])
                contradiction_score = 1.0 - entailment_score

            total_entailment_score += entailment_score

            if contradiction_score > contradiction_threshold:
                ungrounded_claims += 1
            elif entailment_score < entailment_threshold:
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
