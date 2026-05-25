# chronobioticagent/main/agent/llm/prompt_manager.py
"""Prompt Manager for managing and optimizing prompts"""

import json
from pathlib import Path
from string import Template
from typing import Dict, Any, Optional

import yaml


class PromptManager:
    """Manages prompt templates for all agents"""
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or Path(__file__).parent / "prompts"
        self.prompts: Dict[str, Dict] = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompts from files"""
        # Load YAML prompt files
        for prompt_file in self.prompts_dir.glob("*.yaml"):
            with open(prompt_file, 'r') as f:
                prompts = yaml.safe_load(f)
                self.prompts.update(prompts)
        
        # Load JSON prompt files
        for prompt_file in self.prompts_dir.glob("*.json"):
            with open(prompt_file, 'r') as f:
                prompts = json.load(f)
                self.prompts.update(prompts)
    
    def get_prompt(self, prompt_type: str, agent_name: str = None) -> str:
        """Get prompt template by type and agent"""
        # Agent-specific prompt first
        if agent_name and f"{agent_name}_{prompt_type}" in self.prompts:
            return self.prompts[f"{agent_name}_{prompt_type}"]["template"]
        
        # Generic prompt by type
        if prompt_type in self.prompts:
            return self.prompts[prompt_type]["template"]
        
        # Default fallback
        return self._get_default_prompt()
    
    def get_prompt_with_variables(
            self,
            prompt_type: str,
            variables: Dict[str, Any],
            agent_name: str = None
    ) -> str:
        """Get prompt and substitute variables"""
        template = self.get_prompt(prompt_type, agent_name)
        
        # Use string.Template for substitution
        # Convert {var} to ${var} for Template
        template = template.replace("{", "${")
        
        try:
            return Template(template).substitute(**variables)
        except KeyError as e:
            # Handle missing variables
            return Template(template).safe_substitute(**variables)
    
    def _get_default_prompt(self) -> str:
        """Default fallback prompt"""
        return """You are a helpful assistant for chronobiotics research.

Query: {query}

Please provide a helpful response based on your knowledge.
"""
    
    def add_prompt(self, prompt_type: str, template: str, metadata: Dict = None):
        """Add a new prompt template"""
        self.prompts[prompt_type] = {
            "template": template,
            "metadata": metadata or {}
        }
    
    def optimize_prompt(self, prompt_type: str, feedback: Dict) -> str:
        """Optimize prompt based on feedback"""
        current = self.prompts.get(prompt_type, {})
        template = current.get("template", "")
        
        # Apply optimization based on feedback
        if feedback.get("too_long"):
            template = self._truncate_prompt(template)
        if feedback.get("low_accuracy"):
            template = self._add_examples(template)
        if feedback.get("vague_responses"):
            template = self._add_instructions(template)
        
        return template
    
    def _truncate_prompt(self, template: str, max_length: int = 2000) -> str:
        """Truncate prompt to max length"""
        if len(template) > max_length:
            # Keep first and last parts
            first = template[:max_length // 2]
            last = template[-max_length // 2:]
            return first + "\n...[truncated]...\n" + last
        return template
    
    def _add_examples(self, template: str, num_examples: int = 3) -> str:
        """Add few-shot examples to prompt"""
        examples = """
Examples:
Q: What is the half-life of melatonin?
A: Melatonin has a half-life of approximately 3-4 hours in humans.

Q: Does ramelteon affect circadian rhythm?
A: Yes, ramelteon is a melatonin receptor agonist that phase-shifts circadian rhythms.

Q: When should I take melatonin for jet lag?
A: For eastward travel, take melatonin at local bedtime. For westward travel, take upon awakening.
"""
        return template + examples
    
    def _add_instructions(self, template: str) -> str:
        """Add specific instructions to improve responses"""
        instructions = """
Instructions:
- Provide specific, evidence-based answers
- Include confidence levels when uncertain
- Cite sources using [1], [2] format
- Use markdown for formatting
"""
        return template + instructions
