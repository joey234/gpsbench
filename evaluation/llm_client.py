"""
LLM Client for GPSBench Evaluation
Supports OpenAI API, OpenRouter API, and Google Gemini API
"""

import os
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from dotenv import load_dotenv


# Gemini Tier 2 rate limits per model (RPM = Requests Per Minute)
# See: https://ai.google.dev/gemini-api/docs/rate-limits
GEMINI_RATE_LIMITS = {
    # Gemini 2.5 models
    "gemini-2.5-flash": {"rpm": 2000, "tpm": 4_000_000},
    "gemini-2.5-flash-preview": {"rpm": 2000, "tpm": 4_000_000},
    "gemini-2.5-pro": {"rpm": 1000, "tpm": 4_000_000},
    "gemini-2.5-pro-preview": {"rpm": 1000, "tpm": 4_000_000},
    # Gemini 2.0 models
    "gemini-2.0-flash": {"rpm": 2000, "tpm": 4_000_000},
    "gemini-2.0-flash-lite": {"rpm": 4000, "tpm": 4_000_000},
    # Gemini 3 models (preview)
    "gemini-3-pro-preview": {"rpm": 150, "tpm": 1_000_000},
    "gemini-3-flash-preview": {"rpm": 2000, "tpm": 4_000_000},
    # Default for unknown models (conservative)
    "default": {"rpm": 100, "tpm": 1_000_000},
}


class GeminiRateLimiter:
    """
    Thread-safe rate limiter for Gemini API using token bucket algorithm.
    Implements both RPM (requests per minute) limiting and retry with backoff.
    """

    def __init__(self, model: str, rpm_override: Optional[int] = None):
        """
        Initialize rate limiter for a specific Gemini model.

        Args:
            model: Gemini model name (e.g., 'gemini-2.5-flash')
            rpm_override: Override the default RPM limit for this model
        """
        self.model = model
        self.lock = threading.Lock()

        # Get rate limits for this model
        limits = GEMINI_RATE_LIMITS.get(model, GEMINI_RATE_LIMITS["default"])
        self.rpm = rpm_override if rpm_override is not None else limits["rpm"]

        # Token bucket parameters
        self.tokens = self.rpm  # Start with full bucket
        self.last_refill = time.time()
        self.refill_rate = self.rpm / 60.0  # Tokens per second

        # Request tracking for logging
        self.total_requests = 0
        self.rate_limited_count = 0

    def _refill_tokens(self):
        """Refill tokens based on elapsed time (called while holding lock)"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.rpm, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire a token for making a request. Blocks until a token is available
        or timeout is reached.

        Args:
            timeout: Maximum time to wait for a token (seconds)

        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.time()

        while True:
            with self.lock:
                self._refill_tokens()

                if self.tokens >= 1:
                    self.tokens -= 1
                    self.total_requests += 1
                    return True

                # Calculate wait time for next token
                wait_time = (1 - self.tokens) / self.refill_rate

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed + wait_time > timeout:
                return False

            # Wait for token to become available
            time.sleep(min(wait_time, 0.1))  # Poll at most every 100ms

    def record_rate_limit(self):
        """Record that a rate limit error was encountered"""
        with self.lock:
            self.rate_limited_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                "model": self.model,
                "rpm_limit": self.rpm,
                "total_requests": self.total_requests,
                "rate_limited_count": self.rate_limited_count,
                "current_tokens": self.tokens,
            }


# Global rate limiters cache (one per model)
_rate_limiters: Dict[str, GeminiRateLimiter] = {}
_rate_limiters_lock = threading.Lock()


def get_gemini_rate_limiter(model: str, rpm_override: Optional[int] = None) -> GeminiRateLimiter:
    """Get or create a rate limiter for a Gemini model (singleton per model)"""
    with _rate_limiters_lock:
        if model not in _rate_limiters:
            _rate_limiters[model] = GeminiRateLimiter(model, rpm_override)
        return _rate_limiters[model]

# Lazy import for Gemini to avoid requiring it if not used
_genai_client = None

def _get_genai():
    """Lazy load Google GenAI client"""
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai
            _genai_client = genai
        except ImportError:
            raise ImportError(
                "google-genai package not found. Install with: pip install google-genai"
            )
    return _genai_client

# Load environment variables
load_dotenv()


@dataclass
class LLMResponse:
    """Container for LLM response"""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


class LLMClient:
    """
    Unified client for LLM APIs
    Supports OpenAI and OpenRouter
    """

    def __init__(
        self,
        provider: str = "auto",
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 30,
        reasoning_effort: str = "medium",
        gemini_rpm: Optional[int] = None,
        max_retries: int = 5
    ):
        """
        Initialize LLM client

        Args:
            provider: 'openai', 'openrouter', or 'auto' (default: auto-detect from model name)
            model: Model name (e.g., 'gpt-4', 'anthropic/claude-3-opus', 'gpt-5.1')
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            reasoning_effort: For reasoning models, effort level ('none', 'low', 'medium', 'high')
            gemini_rpm: Override RPM (requests per minute) limit for Gemini models
            max_retries: Maximum number of retries on rate limit errors (default: 5)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries

        # Check if this is a reasoning model (gpt-5.x series)
        self.is_reasoning_model = self._is_reasoning_model(model)

        # Auto-detect provider from model name if set to 'auto'
        if provider.lower() == "auto":
            self.provider = self._detect_provider(model)
        else:
            self.provider = provider.lower()

        # Initialize rate limiter for Gemini (will be set after provider detection)
        self.rate_limiter: Optional[GeminiRateLimiter] = None

        # Initialize client based on provider
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        elif self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        elif self.provider == "gemini":
            # Google Gemini API via google-genai package
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")
            genai = _get_genai()
            self.client = genai.Client(api_key=api_key)

            # Initialize rate limiter for Gemini
            self.rate_limiter = get_gemini_rate_limiter(model, gemini_rpm)
            rate_info = GEMINI_RATE_LIMITS.get(model, GEMINI_RATE_LIMITS["default"])
            effective_rpm = gemini_rpm if gemini_rpm is not None else rate_info["rpm"]
            print(f"[Gemini] Rate limiter initialized: {effective_rpm} RPM for {model}")

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _is_reasoning_model(self, model: str) -> bool:
        """
        Check if model is a reasoning model (uses responses API)

        Args:
            model: Model name

        Returns:
            True if reasoning model, False otherwise
        """
        model_lower = model.lower()
        # Reasoning models: gpt-5.x series
        return model_lower.startswith('gpt-5')

    def _detect_provider(self, model: str) -> str:
        """
        Auto-detect provider based on model name format

        Args:
            model: Model name string

        Returns:
            'openai', 'openrouter', or 'gemini'
        """
        model_lower = model.lower()

        # Check for Gemini models first
        # Gemini model names: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro, etc.
        if model_lower.startswith('gemini-'):
            return 'gemini'

        # OpenRouter models typically have a slash format: provider/model
        # Examples: anthropic/claude-3-opus, deepseek/deepseek-chat-v3.1, meta-llama/llama-3-70b

        # Check if model has slash and is a known OpenRouter format
        if '/' in model:
            provider_prefix = model.split('/')[0].lower()

            # Known OpenRouter provider prefixes
            openrouter_providers = {
                'anthropic', 'deepseek', 'meta-llama', 'google',
                'mistralai', 'cohere', 'perplexity', 'nousresearch',
                'qwen', 'phind', 'databricks', '01-ai', 'openchat',
                'teknium', 'undi95', 'gryphe', 'pygmalionai'
            }

            # Special case: openai/ prefix means using OpenAI models via OpenRouter
            if provider_prefix == 'openai':
                return 'openrouter'

            # Check if it's a known OpenRouter provider
            if provider_prefix in openrouter_providers:
                return 'openrouter'

        # Known OpenAI model names (without slash)
        openai_models = {
            'gpt-4', 'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4o', 'gpt-4o-mini',
            'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4-32k', 'gpt-4-1106-preview',
            'o1-preview', 'o1-mini', 'gpt-4.1', 'gpt-4.1-mini',
            'gpt-5', 'gpt-5.1'  # Reasoning models
        }

        # Check for exact match or starts with known OpenAI model
        for openai_model in openai_models:
            if model_lower == openai_model or model_lower.startswith(openai_model + '-'):
                return 'openai'

        # Default to OpenAI for backward compatibility
        return 'openai'

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception is a rate limit error (429)"""
        error_str = str(error).lower()
        # Check for common rate limit error patterns
        return (
            '429' in error_str or
            'rate limit' in error_str or
            'resource_exhausted' in error_str or
            'quota' in error_str or
            'too many requests' in error_str
        )

    def _gemini_generate_with_retry(
        self,
        prompt: str,
        config: Any,
        max_retries: Optional[int] = None
    ) -> Any:
        """
        Generate Gemini response with rate limiting and exponential backoff retry.

        Args:
            prompt: The prompt to send
            config: Gemini generation config
            max_retries: Maximum retry attempts (uses self.max_retries if None)

        Returns:
            Gemini response object

        Raises:
            Exception if all retries fail
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_error = None

        for attempt in range(retries + 1):
            # Acquire rate limit token (blocks if needed)
            if self.rate_limiter:
                if not self.rate_limiter.acquire(timeout=120.0):
                    raise TimeoutError("Timed out waiting for rate limit token")

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
                return response

            except Exception as e:
                last_error = e

                if self._is_rate_limit_error(e):
                    if self.rate_limiter:
                        self.rate_limiter.record_rate_limit()

                    if attempt < retries:
                        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, ...
                        base_delay = 2 ** attempt
                        # Add jitter to prevent thundering herd
                        jitter = base_delay * 0.1 * (0.5 - time.time() % 1)
                        delay = base_delay + jitter

                        print(f"[Gemini] Rate limited (attempt {attempt + 1}/{retries + 1}), "
                              f"retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"[Gemini] Rate limit exceeded after {retries + 1} attempts")
                        raise

                # Non-rate-limit error, don't retry
                raise

        raise last_error

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate response from LLM

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            LLMResponse object with text and metadata
        """
        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            start_time = time.time()

            # Handle Gemini models
            if self.provider == "gemini":
                genai = _get_genai()

                # Build config with system instruction and generation parameters
                config_kwargs = {}
                if system_prompt:
                    config_kwargs['system_instruction'] = system_prompt
                if temp is not None:
                    config_kwargs['temperature'] = temp
                if max_tok is not None:
                    config_kwargs['max_output_tokens'] = max_tok

                # Minimize thinking/reasoning for Gemini models to reduce token usage
                # Gemini 3: Use thinking_level (low for Pro, minimal for Flash)
                # Gemini 2.5/2.0: Use thinking_budget=0 for Flash models
                model_lower = self.model.lower()
                if 'gemini-3' in model_lower:
                    # Gemini 3 uses thinking_level parameter
                    if 'flash' in model_lower:
                        config_kwargs['thinking_config'] = genai.types.ThinkingConfig(thinking_level="minimal")
                    else:
                        # Pro models only support low/high
                        config_kwargs['thinking_config'] = genai.types.ThinkingConfig(thinking_level="low")
                elif ('flash' in model_lower) and ('2.5' in self.model or '2.0' in self.model):
                    # Gemini 2.5/2.0 Flash: Use thinking_budget=0
                    config_kwargs['thinking_config'] = genai.types.ThinkingConfig(thinking_budget=0)

                # Create config if we have any parameters
                config = None
                if config_kwargs:
                    config = genai.types.GenerateContentConfig(**config_kwargs)

                # Generate response with rate limiting and retry
                response = self._gemini_generate_with_retry(prompt, config)

                latency_ms = (time.time() - start_time) * 1000

                # Extract text from response
                text = response.text if hasattr(response, 'text') else str(response)

                # Extract token usage if available
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(usage, 'total_token_count', 0) or 0

                return LLMResponse(
                    text=text.strip() if text else "",
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error=None
                )

            # Handle reasoning models (gpt-5.x) - use chat.completions with max_completion_tokens
            elif self.is_reasoning_model:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                # gpt-5.x requires max_completion_tokens instead of max_tokens
                # gpt-5-mini and gpt-5-nano don't support temperature parameter
                request_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": max_tok,
                }
                # Only add temperature for models that support it (not gpt-5-mini/nano)
                if not any(x in self.model.lower() for x in ['gpt-5-mini', 'gpt-5-nano']):
                    request_kwargs["temperature"] = temp

                response = self.client.chat.completions.create(**request_kwargs)

                latency_ms = (time.time() - start_time) * 1000

                text = response.choices[0].message.content.strip() if response.choices else ""

                # Extract token usage
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0

                return LLMResponse(
                    text=text,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error=None
                )

            else:
                # Standard chat completions API
                messages = []

                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": prompt})

                # Build request kwargs
                request_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tok,
                    "timeout": self.timeout
                }

                # Disable reasoning for OpenRouter to reduce token usage
                if self.provider == "openrouter":
                    request_kwargs["extra_body"] = {
                        "reasoning": {"effort": "none"}
                    }

                response = self.client.chat.completions.create(**request_kwargs)

                latency_ms = (time.time() - start_time) * 1000

                return LLMResponse(
                    text=response.choices[0].message.content.strip(),
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                    latency_ms=latency_ms,
                    error=None
                )

        except Exception as e:
            return LLMResponse(
                text="",
                model=self.model,
                error=str(e)
            )

    def create_batch_file(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        output_file: str = "batch_requests.jsonl",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Create a JSONL file for Batch API (OpenAI or Gemini)

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (same for all)
            output_file: Path to output JSONL file
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Path to created JSONL file
        """
        import json

        tasks = []
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "gemini":
            # Gemini Batch API format
            # Format: {"key": "request-X", "request": {"contents": [...], "generation_config": {...}}}
            for idx, prompt in enumerate(prompts):
                request = {
                    "contents": [{
                        "parts": [{"text": prompt}],
                        "role": "user"
                    }]
                }

                # Add generation config if needed
                generation_config = {}
                if temp is not None:
                    generation_config["temperature"] = temp
                if max_tok is not None:
                    generation_config["maxOutputTokens"] = max_tok

                # Minimize thinking/reasoning for Gemini models to reduce token usage
                model_lower = self.model.lower()
                if 'gemini-3' in model_lower:
                    # Gemini 3 uses thinkingLevel parameter
                    if 'flash' in model_lower:
                        generation_config["thinkingConfig"] = {"thinkingLevel": "minimal"}
                    else:
                        generation_config["thinkingConfig"] = {"thinkingLevel": "low"}
                elif ('flash' in model_lower) and ('2.5' in self.model or '2.0' in self.model):
                    # Gemini 2.5/2.0 Flash: Use thinkingBudget=0
                    generation_config["thinkingConfig"] = {"thinkingBudget": 0}

                if generation_config:
                    request["generation_config"] = generation_config

                # Add system instruction if provided
                if system_prompt:
                    request["system_instruction"] = {
                        "parts": [{"text": system_prompt}]
                    }

                task = {
                    "key": f"request-{idx}",
                    "request": request
                }
                tasks.append(task)

        elif self.is_reasoning_model:
            # OpenAI Reasoning models use /v1/responses endpoint
            for idx, prompt in enumerate(prompts):
                # Combine system and user prompt for responses API
                full_input = prompt
                if system_prompt:
                    full_input = f"{system_prompt}\n\n{prompt}"

                task = {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.model,
                        "input": full_input,
                        "reasoning": {
                            "effort": self.reasoning_effort
                        }
                    }
                }
                tasks.append(task)
        else:
            # OpenAI Chat Completions API format
            for idx, prompt in enumerate(prompts):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                task = {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": max_tok
                    }
                }
                tasks.append(task)

        # Write to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')

        return output_file

    def submit_batch(self, batch_file_path: str, display_name: Optional[str] = None) -> str:
        """
        Upload batch file and create batch job with retry logic for rate limits.

        Args:
            batch_file_path: Path to JSONL batch file
            display_name: Optional display name for the batch job (Gemini only)

        Returns:
            Batch job ID/name
        """
        if self.provider == "gemini":
            genai = _get_genai()

            # Upload file to Gemini File API (with retry)
            uploaded_file = None
            for attempt in range(self.max_retries + 1):
                try:
                    uploaded_file = self.client.files.upload(
                        file=batch_file_path,
                        config=genai.types.UploadFileConfig(
                            display_name=display_name or 'batch-requests',
                            mime_type='application/jsonl'
                        )
                    )
                    break
                except Exception as e:
                    if self._is_rate_limit_error(e) and attempt < self.max_retries:
                        delay = 2 ** attempt
                        print(f"[Gemini Batch] File upload rate limited, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    raise

            # Create batch job with uploaded file (with retry)
            for attempt in range(self.max_retries + 1):
                try:
                    batch_job = self.client.batches.create(
                        model=f"models/{self.model}",
                        src=uploaded_file.name,
                        config={
                            'display_name': display_name or 'gpsbench-batch-job',
                        },
                    )
                    return batch_job.name
                except Exception as e:
                    if self._is_rate_limit_error(e) and attempt < self.max_retries:
                        delay = 2 ** attempt
                        print(f"[Gemini Batch] Batch creation rate limited, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    raise

        elif self.provider == "openai":
            # Upload file
            with open(batch_file_path, "rb") as f:
                batch_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )

            # Reasoning models use /v1/responses, others use /v1/chat/completions
            endpoint = "/v1/responses" if self.is_reasoning_model else "/v1/chat/completions"

            # Create batch job
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint=endpoint,
                completion_window="24h"
            )

            return batch_job.id

        else:
            raise NotImplementedError(f"Batch API is not supported for {self.provider} provider")

    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Check status of a batch job with retry logic for rate limits.

        Args:
            batch_id: Batch job ID/name

        Returns:
            Dictionary with status information
            - For OpenAI: status is one of: validating, failed, in_progress, finalizing, completed, expired, cancelling, cancelled
            - For Gemini: status is one of: JOB_STATE_PENDING, JOB_STATE_RUNNING, JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_CANCELLED, JOB_STATE_EXPIRED
        """
        if self.provider == "gemini":
            # Get batch job with retry
            batch_job = None
            for attempt in range(self.max_retries + 1):
                try:
                    batch_job = self.client.batches.get(name=batch_id)
                    break
                except Exception as e:
                    if self._is_rate_limit_error(e) and attempt < self.max_retries:
                        delay = 2 ** attempt
                        print(f"[Gemini Batch] Status check rate limited, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    raise

            # Map Gemini states to a normalized format
            state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)

            # Normalize status for compatibility
            status_map = {
                'JOB_STATE_PENDING': 'in_progress',
                'JOB_STATE_RUNNING': 'in_progress',
                'JOB_STATE_SUCCEEDED': 'completed',
                'JOB_STATE_FAILED': 'failed',
                'JOB_STATE_CANCELLED': 'cancelled',
                'JOB_STATE_EXPIRED': 'expired',
            }
            normalized_status = status_map.get(state_name, state_name)

            # Get batch stats if available
            # Gemini API uses: total_input_token_count, total_output_token_count, success_count, fail_count
            request_counts = {"total": 0, "completed": 0, "failed": 0}
            if hasattr(batch_job, 'batch_stats') and batch_job.batch_stats:
                stats = batch_job.batch_stats
                # Try different possible attribute names
                total = (
                    getattr(stats, 'total_count', None) or
                    getattr(stats, 'total_request_count', None) or
                    0
                )
                success = (
                    getattr(stats, 'success_count', None) or
                    getattr(stats, 'succeeded_count', None) or
                    0
                )
                failed = (
                    getattr(stats, 'failure_count', None) or
                    getattr(stats, 'fail_count', None) or
                    getattr(stats, 'failed_count', None) or
                    0
                )
                # If total is 0 but we have success/failed, calculate total
                if total == 0 and (success or failed):
                    total = success + failed
                request_counts = {
                    "total": total,
                    "completed": success,
                    "failed": failed,
                }

            return {
                "id": batch_job.name,
                "status": normalized_status,
                "raw_status": state_name,
                "created_at": getattr(batch_job, 'create_time', None),
                "completed_at": getattr(batch_job, 'end_time', None),
                "failed_at": None,
                "request_counts": request_counts,
                "output_file_id": batch_job.dest.file_name if batch_job.dest and hasattr(batch_job.dest, 'file_name') else None,
                "error_file_id": None,
                "error": getattr(batch_job, 'error', None),
            }

        elif self.provider == "openai":
            batch_job = self.client.batches.retrieve(batch_id)

            return {
                "id": batch_job.id,
                "status": batch_job.status,
                "raw_status": batch_job.status,
                "created_at": batch_job.created_at,
                "completed_at": getattr(batch_job, 'completed_at', None),
                "failed_at": getattr(batch_job, 'failed_at', None),
                "request_counts": {
                    "total": batch_job.request_counts.total,
                    "completed": batch_job.request_counts.completed,
                    "failed": batch_job.request_counts.failed
                },
                "output_file_id": getattr(batch_job, 'output_file_id', None),
                "error_file_id": getattr(batch_job, 'error_file_id', None)
            }

        else:
            raise NotImplementedError(f"Batch API is not supported for {self.provider} provider")

    def retrieve_batch_results(self, batch_id: str, output_file: str = "batch_results.jsonl", expected_count: Optional[int] = None) -> List[LLMResponse]:
        """
        Retrieve and parse results from completed batch job with retry logic.

        Args:
            batch_id: Batch job ID/name
            output_file: Path to save results JSONL file
            expected_count: Expected number of results (optional, for validation)

        Returns:
            List of LLMResponse objects in original order
        """
        import json

        results_dict = {}

        if self.provider == "gemini":
            # Get batch job with retry
            batch_job = None
            for attempt in range(self.max_retries + 1):
                try:
                    batch_job = self.client.batches.get(name=batch_id)
                    break
                except Exception as e:
                    if self._is_rate_limit_error(e) and attempt < self.max_retries:
                        delay = 2 ** attempt
                        print(f"[Gemini Batch] Get batch rate limited, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    raise

            state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
            if state_name != 'JOB_STATE_SUCCEEDED':
                error_info = ""
                if hasattr(batch_job, 'error') and batch_job.error:
                    error_info = f" Error: {batch_job.error}"
                raise ValueError(f"Batch job not completed. Status: {state_name}.{error_info}")

            # Check for results - either file or inline
            if batch_job.dest and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                # Results are in a file - download with retry
                result_file_name = batch_job.dest.file_name
                file_content_bytes = None
                for attempt in range(self.max_retries + 1):
                    try:
                        file_content_bytes = self.client.files.download(file=result_file_name)
                        break
                    except Exception as e:
                        if self._is_rate_limit_error(e) and attempt < self.max_retries:
                            delay = 2 ** attempt
                            print(f"[Gemini Batch] File download rate limited, retrying in {delay}s...")
                            time.sleep(delay)
                            continue
                        raise
                file_content = file_content_bytes.decode('utf-8')

                # Save to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(file_content)

                # Parse JSONL results
                for line in file_content.splitlines():
                    if not line.strip():
                        continue
                    result = json.loads(line)

                    # Extract key (format: "request-{idx}")
                    key = result.get('key', '')
                    try:
                        idx = int(key.split('-')[-1])
                    except (ValueError, IndexError):
                        continue

                    if 'response' in result and result['response']:
                        response = result['response']
                        # Extract text from Gemini response format
                        text = ""
                        if 'candidates' in response and response['candidates']:
                            candidate = response['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                for part in candidate['content']['parts']:
                                    if 'text' in part:
                                        text = part['text']
                                        break

                        # Extract token usage
                        usage = response.get('usageMetadata', {})
                        results_dict[idx] = LLMResponse(
                            text=text.strip() if text else "",
                            model=self.model,
                            prompt_tokens=usage.get('promptTokenCount', 0) or 0,
                            completion_tokens=usage.get('candidatesTokenCount', 0) or 0,
                            total_tokens=usage.get('totalTokenCount', 0) or 0,
                            error=None
                        )
                    elif 'error' in result:
                        results_dict[idx] = LLMResponse(
                            text="",
                            model=self.model,
                            error=str(result['error'])
                        )

            elif batch_job.dest and hasattr(batch_job.dest, 'inlined_responses') and batch_job.dest.inlined_responses:
                # Results are inline
                for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                    if inline_response.response:
                        # Extract text using the .text property if available
                        try:
                            text = inline_response.response.text
                        except AttributeError:
                            # Fallback to manual extraction
                            text = ""
                            if hasattr(inline_response.response, 'candidates') and inline_response.response.candidates:
                                candidate = inline_response.response.candidates[0]
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text'):
                                            text = part.text
                                            break

                        # Extract token usage
                        usage_metadata = getattr(inline_response.response, 'usage_metadata', None)
                        prompt_tokens = 0
                        completion_tokens = 0
                        total_tokens = 0
                        if usage_metadata:
                            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
                            completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
                            total_tokens = getattr(usage_metadata, 'total_token_count', 0) or 0

                        results_dict[i] = LLMResponse(
                            text=text.strip() if text else "",
                            model=self.model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            error=None
                        )
                    elif inline_response.error:
                        results_dict[i] = LLMResponse(
                            text="",
                            model=self.model,
                            error=str(inline_response.error)
                        )
            else:
                raise ValueError(f"Batch {batch_id} completed but has no results (neither file nor inline).")

        elif self.provider == "openai":
            # Get batch job
            batch_job = self.client.batches.retrieve(batch_id)

            if batch_job.status != "completed":
                raise ValueError(f"Batch job not completed. Status: {batch_job.status}")

            # Download results
            result_file_id = batch_job.output_file_id

            # Check if batch has results
            if result_file_id is None:
                error_msg = f"Batch {batch_id} completed but has no output file."
                if hasattr(batch_job, 'request_counts'):
                    error_msg += f"\nRequest counts: completed={batch_job.request_counts.completed}, failed={batch_job.request_counts.failed}, total={batch_job.request_counts.total}"
                if hasattr(batch_job, 'errors') and batch_job.errors:
                    error_msg += f"\nBatch errors: {batch_job.errors}"

                # Download and show error file if available
                if batch_job.error_file_id:
                    try:
                        error_content = self.client.files.content(batch_job.error_file_id).content
                        error_text = error_content.decode('utf-8')
                        error_msg += f"\n\n--- Error File Contents ({batch_job.error_file_id}) ---\n{error_text[:2000]}"
                        if len(error_text) > 2000:
                            error_msg += f"\n... (truncated, {len(error_text)} total chars)"
                    except Exception as e:
                        error_msg += f"\nError file ID: {batch_job.error_file_id} (failed to download: {e})"

                raise ValueError(error_msg)

            result_content = self.client.files.content(result_file_id).content

            # Save to file
            with open(output_file, 'wb') as f:
                f.write(result_content)

            # Parse results
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line.strip())
                    custom_id = result['custom_id']
                    # Extract index from custom_id (format: "request-{idx}")
                    idx = int(custom_id.split('-')[-1])

                    if result['response']['status_code'] == 200:
                        body = result['response']['body']

                        # Handle both responses API and chat completions API formats
                        if 'output' in body:
                            # Responses API format (reasoning models)
                            output = body['output']
                            if isinstance(output, list):
                                text = ""
                                for item in output:
                                    if item.get('type') == 'message' and 'content' in item:
                                        for content_item in item['content']:
                                            if content_item.get('type') == 'output_text':
                                                text = content_item.get('text', '')
                                                break
                                        if text:
                                            break
                                if not text:
                                    text = str(output)
                            elif isinstance(output, str):
                                text = output
                            else:
                                text = str(output)
                            text = text.strip()
                        elif 'choices' in body:
                            # Chat Completions API format
                            text = body['choices'][0]['message']['content'].strip()
                        else:
                            text = str(body)

                        # Extract token usage
                        usage = body.get('usage', {})
                        results_dict[idx] = LLMResponse(
                            text=text,
                            model=self.model,
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            completion_tokens=usage.get('completion_tokens', 0),
                            total_tokens=usage.get('total_tokens', 0),
                            error=None
                        )
                    else:
                        error_msg = result['response'].get('body', {}).get('error', {}).get('message', 'Unknown error')
                        results_dict[idx] = LLMResponse(
                            text="",
                            model=self.model,
                            error=error_msg
                        )

        else:
            raise NotImplementedError(f"Batch API is not supported for {self.provider} provider")

        # Determine max index from either expected_count or results
        if expected_count is not None:
            max_idx = expected_count - 1
        else:
            max_idx = max(results_dict.keys()) if results_dict else -1

        # Convert to ordered list
        results = []
        for i in range(max_idx + 1):
            if i in results_dict:
                results.append(results_dict[i])
            else:
                # Missing result
                results.append(LLMResponse(
                    text="",
                    model=self.model,
                    error="Result not found in batch output"
                ))

        return results

    def retry_failed_batch_samples(
        self,
        failed_indices: List[int],
        original_prompts: List[str],
        system_prompt: Optional[str] = None,
        output_file: str = "batch_retry.jsonl",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Create and submit a batch job to retry failed samples

        Args:
            failed_indices: List of indices that failed
            original_prompts: Full list of original prompts
            system_prompt: Optional system prompt
            output_file: Path to output JSONL file
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Batch job ID for retry
        """
        import json

        # Create retry prompts with original indices
        retry_prompts = []
        index_mapping = []  # Map retry index to original index

        for failed_idx in sorted(failed_indices):
            if failed_idx < len(original_prompts):
                retry_prompts.append(original_prompts[failed_idx])
                index_mapping.append(failed_idx)

        if not retry_prompts:
            raise ValueError("No valid failed samples to retry")

        # Create batch file with custom_ids that preserve original indices
        tasks = []
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "gemini":
            # Gemini Batch API format
            for retry_idx, original_idx in enumerate(index_mapping):
                prompt = retry_prompts[retry_idx]
                request = {
                    "contents": [{
                        "parts": [{"text": prompt}],
                        "role": "user"
                    }]
                }

                # Add generation config if needed
                generation_config = {}
                if temp is not None:
                    generation_config["temperature"] = temp
                if max_tok is not None:
                    generation_config["maxOutputTokens"] = max_tok

                # Minimize thinking/reasoning for Gemini models to reduce token usage
                model_lower = self.model.lower()
                if 'gemini-3' in model_lower:
                    # Gemini 3 uses thinkingLevel parameter
                    if 'flash' in model_lower:
                        generation_config["thinkingConfig"] = {"thinkingLevel": "minimal"}
                    else:
                        generation_config["thinkingConfig"] = {"thinkingLevel": "low"}
                elif ('flash' in model_lower) and ('2.5' in self.model or '2.0' in self.model):
                    # Gemini 2.5/2.0 Flash: Use thinkingBudget=0
                    generation_config["thinkingConfig"] = {"thinkingBudget": 0}

                if generation_config:
                    request["generation_config"] = generation_config

                # Add system instruction if provided
                if system_prompt:
                    request["system_instruction"] = {
                        "parts": [{"text": system_prompt}]
                    }

                task = {
                    "key": f"request-{original_idx}",  # Use original index
                    "request": request
                }
                tasks.append(task)

        elif self.is_reasoning_model:
            # OpenAI Reasoning models use /v1/responses endpoint
            for retry_idx, original_idx in enumerate(index_mapping):
                prompt = retry_prompts[retry_idx]
                # Combine system and user prompt for responses API
                full_input = prompt
                if system_prompt:
                    full_input = f"{system_prompt}\n\n{prompt}"

                task = {
                    "custom_id": f"request-{original_idx}",  # Use original index
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.model,
                        "input": full_input,
                        "reasoning": {
                            "effort": self.reasoning_effort
                        }
                    }
                }
                tasks.append(task)
        else:
            # OpenAI Chat Completions API format
            for retry_idx, original_idx in enumerate(index_mapping):
                prompt = retry_prompts[retry_idx]
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                task = {
                    "custom_id": f"request-{original_idx}",  # Use original index
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": max_tok
                    }
                }
                tasks.append(task)

        # Write to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')

        # Submit batch
        batch_id = self.submit_batch(output_file)

        return batch_id

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        delay: float = 0.0,
        max_workers: int = 10
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts concurrently (using threading, not Batch API)

        For OpenAI Batch API (50% cost savings, 24h processing), use:
        1. create_batch_file()
        2. submit_batch()
        3. check_batch_status() (poll until complete)
        4. retrieve_batch_results()

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (same for all)
            delay: Delay between batch submissions (not between individual requests)
            max_workers: Maximum number of concurrent requests (default: 10)

        Returns:
            List of LLMResponse objects in the same order as input prompts
        """
        from tqdm import tqdm

        if not prompts:
            return []

        # For single prompt, use regular generate
        if len(prompts) == 1:
            return [self.generate(prompts[0], system_prompt)]

        # Store results with their original indices to maintain order
        results = [None] * len(prompts)

        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.generate, prompt, system_prompt): idx
                for idx, prompt in enumerate(prompts)
            }

            # Collect results as they complete with progress bar
            with tqdm(total=len(prompts), desc="Processing concurrent requests") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        response = future.result()
                        results[idx] = response
                    except Exception as e:
                        # Create error response for failed requests
                        results[idx] = LLMResponse(
                            text="",
                            model=self.model,
                            error=str(e)
                        )
                    pbar.update(1)

        # Optional delay after batch (for rate limiting between batches)
        if delay > 0:
            time.sleep(delay)

        return results

    def get_rate_limiter_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get rate limiter statistics (Gemini only).

        Returns:
            Dictionary with rate limiter stats, or None if not using Gemini
        """
        if self.rate_limiter:
            return self.rate_limiter.get_stats()
        return None

    def __repr__(self):
        if self.rate_limiter:
            stats = self.rate_limiter.get_stats()
            return (f"LLMClient(provider={self.provider}, model={self.model}, "
                    f"rpm={stats['rpm_limit']}, requests={stats['total_requests']})")
        return f"LLMClient(provider={self.provider}, model={self.model})"


def get_available_models() -> Dict[str, List[str]]:
    """
    Get commonly used models for each provider

    Returns:
        Dictionary mapping provider to list of model names
    """
    return {
        "openai": [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "o1-preview",
            "o1-mini"
        ],
        "gemini": [
            # Gemini 3 models (latest)
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            # Gemini 2.5 models
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            # Gemini 2.0 models
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            # Gemini 1.5 models
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ],
        "openrouter": [
            # OpenAI models via OpenRouter
            "openai/gpt-4",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",

            # Anthropic models
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "anthropic/claude-2",

            # Google models
            "google/gemini-pro",
            "google/gemini-pro-1.5",

            # Meta models
            "meta-llama/llama-3-70b-instruct",
            "meta-llama/llama-3-8b-instruct",

            # Other models
            "mistralai/mistral-large",
            "mistralai/mistral-medium",
        ]
    }


# Example usage
if __name__ == "__main__":
    # Test OpenAI
    try:
        client = LLMClient(provider="openai", model="gpt-3.5-turbo")
        response = client.generate("What is 2+2?")
        print(f"OpenAI Response: {response.text}")
        print(f"Tokens: {response.total_tokens}, Latency: {response.latency_ms:.0f}ms")
    except Exception as e:
        print(f"OpenAI test failed: {e}")

    # Test OpenRouter
    try:
        client = LLMClient(provider="openrouter", model="openai/gpt-3.5-turbo")
        response = client.generate("What is 2+2?")
        print(f"\nOpenRouter Response: {response.text}")
        print(f"Tokens: {response.total_tokens}, Latency: {response.latency_ms:.0f}ms")
    except Exception as e:
        print(f"OpenRouter test failed: {e}")
