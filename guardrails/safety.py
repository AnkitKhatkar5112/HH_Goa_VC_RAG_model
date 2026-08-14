"""Unsafe/inappropriate input content safety check.

Two-pass approach:
1. Fast keyword blocklist (multilingual) — catches obvious cases.
2. LLM-based safety classification for subtler cases (optional).
"""

from __future__ import annotations

import re

from guardrails.schemas import GuardrailResult

# Curated blocklist — English, Hindi, Tamil common terms
# This is intentionally kept short and focused; a production system
# would use a more comprehensive list or a dedicated moderation API.
UNSAFE_PATTERNS = [
    # English patterns
    r"\b(kill|murder|assassinate|bomb|terrorist|hack\s+into)\b",
    r"\b(how\s+to\s+make\s+(a\s+)?bomb|how\s+to\s+poison)\b",
    r"\b(suicide|self[- ]harm)\b",
    r"\b(child\s+(porn|abuse|exploitation))\b",
    r"\b(illegal\s+drugs?|how\s+to\s+(buy|make)\s+(meth|cocaine|heroin))\b",
    # Hindi patterns (Devanagari)
    r"(बम\s+बनान|हत्या\s+कर|आतंकवाद)",
    r"(ड्रग्स\s+कैसे|नशीली\s+दवा)",
    # Tamil patterns
    r"(குண்டு\s+செய்|கொலை\s+செய்)",
    # Prompt injection patterns
    r"(ignore\s+(previous|all)\s+(instructions?|prompts?))",
    r"(you\s+are\s+now|pretend\s+you\s+are|act\s+as\s+if)",
    r"(system\s+prompt|reveal\s+your\s+(instructions?|prompt))",
    r"(jailbreak|DAN\s+mode)",
]

# Compile patterns for performance
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in UNSAFE_PATTERNS]

REFUSAL_MESSAGES = {
    "en": "I cannot process this request as it may contain inappropriate or unsafe content.",
    "hi": "मैं इस अनुरोध को संसाधित नहीं कर सकता क्योंकि इसमें अनुचित या असुरक्षित सामग्री हो सकती है।",
    "ta": "இந்தக் கோரிக்கையை செயலாக்க முடியாது, ஏனெனில் இதில் பொருத்தமற்ற அல்லது பாதுகாப்பற்ற உள்ளடக்கம் இருக்கலாம்.",
}


def check_safety(
    query: str,
    query_language: str = "en",
) -> GuardrailResult:
    """Check if a query contains unsafe or inappropriate content.

    Args:
        query: The transcribed/input query text.
        query_language: Language for the refusal message.

    Returns:
        GuardrailResult indicating whether the query passed safety check.
    """
    query_lower = query.lower().strip()

    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(query_lower)
        if match:
            refusal = REFUSAL_MESSAGES.get(query_language, REFUSAL_MESSAGES["en"])
            return GuardrailResult(
                passed=False,
                reason=refusal,
                category="unsafe",
                confidence=0.95,
            )

    # Passed all checks
    return GuardrailResult(
        passed=True,
        reason=None,
        category="unsafe",
        confidence=0.0,
    )


async def check_safety_with_llm(
    query: str,
    llm_client,
    query_language: str = "en",
) -> GuardrailResult:
    """Enhanced safety check using LLM classification.

    Only called if keyword check passes but we want extra confidence.

    Args:
        query: The query text.
        llm_client: The LLM client for classification.
        query_language: Language for refusal message.

    Returns:
        GuardrailResult from LLM classification.
    """
    prompt = (
        "You are a content safety classifier. Determine if the following user query "
        "asks for harmful, illegal, dangerous, or inappropriate content.\n\n"
        f"Query: {query}\n\n"
        "Respond with ONLY 'SAFE' or 'UNSAFE'. Nothing else."
    )

    try:
        response = await llm_client.generate_text(prompt, max_tokens=10)
        is_safe = "SAFE" in response.upper() and "UNSAFE" not in response.upper()

        if is_safe:
            return GuardrailResult(passed=True, reason=None, category="unsafe", confidence=0.0)
        else:
            refusal = REFUSAL_MESSAGES.get(query_language, REFUSAL_MESSAGES["en"])
            return GuardrailResult(passed=False, reason=refusal, category="unsafe", confidence=0.85)
    except Exception:
        # On LLM failure, pass through (fail open for safety classifier)
        return GuardrailResult(passed=True, reason=None, category="unsafe", confidence=0.0)
