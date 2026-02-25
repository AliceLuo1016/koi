"""Context window management and compaction for koi agent."""

import asyncio
import json
from typing import List, Dict, Any
import tiktoken

from .llm import LLMClient


class ContextCompactor:
    """Handle context window management and message compaction."""
    
    def __init__(self, llm_client: LLMClient, context_window: int):
        """Initialize compactor with LLM client and context window size."""
        self.llm_client = llm_client
        self.context_window = context_window
        
        # Initialize tokenizer for rough token estimation
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            # Fallback to a general tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for a list of messages."""
        total_tokens = 0
        
        for message in messages:
            # Count tokens for role
            total_tokens += 4  # Rough estimate for role and formatting
            
            # Count tokens for content
            if message.get("content"):
                total_tokens += len(self.tokenizer.encode(str(message["content"])))
            
            # Count tokens for tool calls
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    total_tokens += len(self.tokenizer.encode(json.dumps(tool_call)))
        
        return total_tokens
    
    def needs_compaction(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if messages exceed 70% of context window."""
        estimated_tokens = self.estimate_tokens(messages)
        threshold = self.context_window * 0.7
        
        return estimated_tokens > threshold
    
    def _safe_split_index(self, messages: List[Dict[str, Any]], split_index: int) -> int:
        """Adjust split index so it never lands between a tool call and its results.

        If the message at split_index is a tool result, walk backward to include
        the preceding assistant message with tool_calls on the keep side.
        """
        while split_index > 1 and messages[split_index].get("role") == "tool":
            split_index -= 1
        return split_index

    async def compact_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compact messages by summarizing the oldest 40%."""
        if len(messages) <= 3:  # Keep at least system prompt and recent messages
            return messages

        # Calculate split point (40% of messages)
        split_index = max(1, int(len(messages) * 0.4))  # Keep at least the system message

        # Ensure we don't split in the middle of a tool call/result pair
        split_index = self._safe_split_index(messages, split_index)

        # Separate messages to compact vs keep
        to_compact = messages[:split_index]
        to_keep = messages[split_index:]

        # Create summary of messages to compact
        try:
            summary = await self._create_summary(to_compact)

            # Create summary message
            summary_message = {
                "role": "system",
                "content": f"[Previous conversation summary: {summary}]"
            }

            # Return compacted conversation
            return [summary_message] + to_keep

        except asyncio.CancelledError:
            raise

        except Exception as e:
            # If compaction fails, just truncate
            print(f"Warning: Compaction failed ({e}), truncating instead")
            fallback_index = len(messages) - int(len(messages) * 0.6)
            fallback_index = self._safe_split_index(messages, fallback_index)
            return messages[fallback_index:]
    
    async def _create_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Create a concise summary of messages using LLM."""
        # Prepare messages for summarization
        conversation_text = self._messages_to_text(messages)
        
        summary_prompt = f"""Please provide a concise summary of this conversation, focusing on:
- Key topics discussed
- Important decisions or conclusions
- Relevant context for future reference

Conversation to summarize:
{conversation_text}

Summary:"""
        
        summary_messages = [
            {"role": "user", "content": summary_prompt}
        ]
        
        # Get summary from LLM
        response = await self.llm_client.chat(summary_messages)
        
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"].strip()
        else:
            return "Previous conversation context (summary unavailable)"
    
    def _messages_to_text(self, messages: List[Dict[str, Any]]) -> str:
        """Convert messages to readable text for summarization."""
        text_parts = []
        
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            if role == "system":
                text_parts.append(f"[System]: {content}")
            elif role == "user":
                text_parts.append(f"User: {content}")
            elif role == "assistant":
                if content:
                    text_parts.append(f"Assistant: {content}")
                
                # Handle tool calls
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        func_name = tool_call.get("function", {}).get("name", "unknown")
                        text_parts.append(f"[Called tool: {func_name}]")
            
            elif role == "tool":
                tool_content = content[:200] + "..." if len(content) > 200 else content
                text_parts.append(f"[Tool result]: {tool_content}")
        
        return "\n".join(text_parts)
    
    def get_context_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about current context usage."""
        estimated_tokens = self.estimate_tokens(messages)
        usage_percent = (estimated_tokens / self.context_window) * 100
        
        return {
            "estimated_tokens": estimated_tokens,
            "context_window": self.context_window,
            "usage_percent": round(usage_percent, 1),
            "needs_compaction": self.needs_compaction(messages),
            "message_count": len(messages)
        }