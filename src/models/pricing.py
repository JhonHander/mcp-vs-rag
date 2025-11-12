"""
Pricing module for LLM cost calculation.

This module contains pricing information for different LLM models and utilities
to calculate costs based on token usage.

Pricing is per 1 million tokens (as of November 2025).
Update MODEL_PRICING when providers change their rates.
"""

# Pricing per 1M tokens (USD)
# Source: OpenAI and Google AI pricing pages (November 2025)
MODEL_PRICING = {
    "gpt-5": {
        "input_cost_per_1m": 5.00,   # USD per 1M input tokens
        "output_cost_per_1m": 15.00,  # USD per 1M output tokens
        "notes": "OpenAI GPT-5 pricing (released August 2025)"
    },
    "gemini-2.5-pro": {
        "input_cost_per_1m": 1.25,    # USD per 1M input tokens
        "output_cost_per_1m": 5.00,   # USD per 1M output tokens
        "notes": "Google Gemini 2.5 Pro pricing"
    }
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> dict:
    """
    Calculate the cost in USD for an LLM API call based on token usage.
    
    Args:
        model_name: Name of the model ("gpt-5" or "gemini-2.5-pro")
        input_tokens: Number of input (prompt) tokens
        output_tokens: Number of output (completion) tokens
    
    Returns:
        Dictionary containing:
            - model: Model name used
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens
            - total_tokens: Sum of input and output tokens
            - input_cost_usd: Cost for input tokens in USD
            - output_cost_usd: Cost for output tokens in USD
            - total_cost_usd: Total cost in USD
            - error: Error message if pricing data not available (optional)
    
    Example:
        >>> cost = calculate_cost("gpt-5", 1000, 500)
        >>> print(cost)
        {
            'model': 'gpt-5',
            'input_tokens': 1000,
            'output_tokens': 500,
            'total_tokens': 1500,
            'input_cost_usd': 0.005,
            'output_cost_usd': 0.0075,
            'total_cost_usd': 0.0125
        }
    """
    # Get pricing data for the model
    pricing = MODEL_PRICING.get(model_name)
    
    if not pricing:
        return {
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "error": f"No pricing data available for model: {model_name}"
        }
    
    # Calculate costs (price per million / 1,000,000 * tokens)
    input_cost = (pricing["input_cost_per_1m"] / 1_000_000) * input_tokens
    output_cost = (pricing["output_cost_per_1m"] / 1_000_000) * output_tokens
    total_cost = input_cost + output_cost
    
    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_cost, 8)
    }


def get_model_pricing_info(model_name: str = None) -> dict:
    """
    Get pricing information for a specific model or all models.
    
    Args:
        model_name: Optional model name. If None, returns all pricing data.
    
    Returns:
        Dictionary with pricing information
    
    Example:
        >>> info = get_model_pricing_info("gpt-5")
        >>> print(info)
        {
            'input_cost_per_1m': 5.0,
            'output_cost_per_1m': 15.0,
            'notes': 'OpenAI GPT-5 pricing (released August 2025)'
        }
    """
    if model_name:
        return MODEL_PRICING.get(model_name, {})
    return MODEL_PRICING
