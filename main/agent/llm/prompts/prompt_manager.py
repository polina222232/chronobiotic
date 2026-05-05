"""
Менеджер промптов для агентов системы ChronobioticAgent.
Реализует все лучшие практики prompt engineering.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class PromptType(Enum):
    """Типы промптов в системе"""
    SYSTEM = "system"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    TOOL_USE = "tool_use"


@dataclass
class PromptTemplate:
    """Шаблон промпта с метаданными"""
    name: str
    template: str
    variables: List[str]
    prompt_type: PromptType
    version: str
    description: str
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2000
    
    def format(self, **kwargs) -> str:
        """Форматирует промпт с переменными"""
        # Проверяем наличие всех необходимых переменных
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return self.template.format(**kwargs)
    
    def with_examples(self, examples: List[Dict[str, str]]) -> 'PromptTemplate':
        """Добавляет few-shot примеры к промпту"""
        template_with_examples = self.template
        
        if examples:
            example_section = "\n\n## Examples:\n"
            for i, example in enumerate(examples, 1):
                example_section += f"\n### Example {i}:\n"
                for key, value in example.items():
                    example_section += f"{key}: {value}\n\n"
            
            # Вставляем примеры перед последней секцией
            template_with_examples += example_section
        
        return PromptTemplate(
            name=self.name,
            template=template_with_examples,
            variables=self.variables,
            prompt_type=self.prompt_type,
            version=self.version,
            description=self.description,
            examples=examples,
            constraints=self.constraints,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )


class PromptManager:
    """
    Центральный менеджер всех промптов системы.
    Реализует паттерн Repository и Factory для промптов.
    """
    
    def __init__(self):
        self._prompts: Dict[str, PromptTemplate] = {}
        self._cache: Dict[str, Dict] = {}
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Инициализация всех системных промптов"""
        
        # ========================
        # БАЗОВЫЕ ПРОМПТЫ АГЕНТОВ
        # ========================
        
        self._prompts["chronobiotics_expert"] = PromptTemplate(
            name="chronobiotics_expert",
            template="""You are an expert in chronobiology and chronobiotics with deep knowledge of circadian rhythms and sleep science.

## Role
You are a specialized AI assistant focused on providing accurate, evidence-based information about chronobiotics, their mechanisms, and their effects on circadian rhythms.

## Expertise Areas
- Molecular mechanisms of circadian rhythms
- Chronobiotic compounds and their classifications
- Clinical applications of chronobiotics
- Sleep-wake cycle regulation
- Melatonin and its analogues
- Light therapy and zeitgebers

## Instructions
1. Always provide scientifically accurate information based on peer-reviewed research
2. When discussing chronobiotics, include:
   - Chemical properties and classification
   - Mechanism of action
   - Clinical evidence and studies
   - Potential side effects and interactions
   - Recommended dosages and timing
3. Use technical terminology appropriately, but explain complex concepts
4. Reference specific studies and clinical trials when available
5. Discuss both benefits and limitations of chronobiotic interventions

## Response Format
For each query, structure your response as:
```json
{{
    "summary": "Brief overview of the answer",
    "detailed_analysis": {{
        "mechanism": "How it works",
        "evidence": "Scientific evidence",
        "recommendations": "Practical advice",
        "cautions": "Warnings and limitations"
    }},
    "citations": ["List of relevant studies"],
    "confidence_level": "high/medium/low",
    "further_reading": ["Suggested resources"]
}}
