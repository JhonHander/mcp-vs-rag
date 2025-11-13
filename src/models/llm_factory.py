from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import UsageMetadataCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()


class TrackedLLM:
    """
    Wrapper that adds automatic cost tracking to any LangChain chat model.
    
    This class wraps ChatOpenAI or ChatGoogleGenerativeAI instances and automatically
    captures token usage and calculates costs for each LLM invocation.
    
    The wrapper is transparent - it passes through all the same arguments and returns
    the same response format, but additionally returns cost information.
    
    Attributes:
        model: The underlying LangChain chat model (ChatOpenAI or ChatGoogleGenerativeAI)
        model_name: Name of the model for cost calculation ("gpt-5" or "gemini-2.5-pro")
    
    Example:
        >>> llm = TrackedLLM(ChatOpenAI(model="gpt-5"), "gpt-5")
        >>> response, cost = llm.invoke("What is AI?")
        >>> print(f"Cost: ${cost['total_cost_usd']:.6f}")
        Cost: $0.000234
    """
    
    def __init__(self, model, model_name: str):
        """
        Initialize the TrackedLLM wrapper.
        
        Args:
            model: LangChain chat model instance (ChatOpenAI or ChatGoogleGenerativeAI)
            model_name: Model name for pricing lookup ("gpt-5" or "gemini-2.5-pro")
        """
        self.model = model
        self.model_name = model_name
    
    def invoke(self, *args, **kwargs):
        """
        Invoke the LLM and capture cost information.
        
        This method:
        1. Creates a UsageMetadataCallbackHandler to capture token usage
        2. Adds the callback to the model's configuration
        3. Invokes the underlying model with the callback attached
        4. Extracts token usage from the response
        5. Calculates cost using the pricing module
        6. Returns both the response and cost information
        
        Args:
            *args: Positional arguments passed to the underlying model's invoke()
            **kwargs: Keyword arguments passed to the underlying model's invoke()
        
        Returns:
            tuple: (response, cost_info)
                - response: AIMessage object from the LLM
                - cost_info: Dictionary with token counts and cost breakdown
        
        Example:
            >>> response, cost = llm.invoke("Hello, world!")
            >>> print(response.content)
            "Hello! How can I assist you today?"
            >>> print(cost)
            {
                'model': 'gpt-5',
                'input_tokens': 8,
                'output_tokens': 10,
                'total_tokens': 18,
                'input_cost_usd': 0.00004,
                'output_cost_usd': 0.00015,
                'total_cost_usd': 0.00019
            }
        """
        # Create callback to capture token usage
        callback = UsageMetadataCallbackHandler()
        
        # Add callback to the configuration
        config = kwargs.get("config", {})
        callbacks = config.get("callbacks", [])
        callbacks.append(callback)
        config["callbacks"] = callbacks
        kwargs["config"] = config
        
        # Invoke the underlying model with the callback
        response = self.model.invoke(*args, **kwargs)
        
        # Extract token usage from the response
        # Both OpenAI and Google models return usage_metadata
        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        # Calculate costs using the pricing module
        from .pricing import calculate_cost
        cost_info = calculate_cost(
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        # Return both response and cost information
        return response, cost_info


def create_llm(model_name: str):
    """
    Factory function to create LLM instances with automatic cost tracking.

    This function creates a base LangChain chat model (ChatOpenAI or ChatGoogleGenerativeAI)
    and wraps it with TrackedLLM to enable automatic token usage capture and cost calculation.

    Supported models:
    - "gpt-5": OpenAI GPT-5 (released August 2025)
    - "gemini-2.5-pro": Google Gemini 2.5 Pro
    
    Returns:
        TrackedLLM: Wrapper that automatically captures tokens and costs on each invoke()
    
    Raises:
        ValueError: If an unsupported model name is provided
    
    Example:
        >>> llm = create_llm("gpt-5")
        >>> response, cost = llm.invoke("What is AI?")
        >>> print(f"Answer: {response.content}")
        >>> print(f"Cost: ${cost['total_cost_usd']:.6f}")
        Answer: Artificial Intelligence is...
        Cost: $0.000234
    """

    # Create the base model based on the model name
    # Optimized settings: shorter timeouts, max_tokens limit, streaming disabled
    if model_name == "gpt-5":
        base_model = ChatOpenAI(
            model="gpt-5",
            temperature=0,
            request_timeout=60,  # 60 second timeout to avoid hanging (increased from 30s)
            max_retries=3,  # Retry 3 times to handle network issues
            streaming=False,  # Disable streaming for faster batch responses
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_name == "gemini-2.5-pro":
        base_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            # max_output_tokens=500,  # Limit response length
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Use 'gpt-5' or 'gemini-2.5-pro'")
    
    # Wrap the base model with TrackedLLM for automatic cost tracking
    return TrackedLLM(base_model, model_name)


def get_available_models():
    """Get list of available models."""
    return ["gpt-5", "gemini-2.5-pro"]
