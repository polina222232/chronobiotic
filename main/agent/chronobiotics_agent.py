
from agent_core import BaseAgent, AgentType, Message
from typing import Dict, Any, List
import json


class ChronobioticsAgent(BaseAgent):
    """Главный интегрирующий агент для исследования хронобиотиков."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.CHRONOBIOTICS)
        self.research_context = {
            "current_hypothesis": None,
            "experiments": [],
            "findings": [],
            "literature_references": [],
            "temporal_patterns": {}
        }
        self.active_tasks: Dict[str, Dict] = {}
    
    async def _setup_handlers(self):
        """Настройка обработчиков для комплексных исследовательских задач."""
        self.handlers = {
            "initiate_research": self.handle_initiate_research,
            "analyze_compound": self.handle_analyze_compound,
            "generate_report": self.handle_generate_report,
            "data_analysis_result": self.handle_data_result,
            "literature_review_result": self.handle_literature_result,
        }
    
    async def handle_initiate_research(self, message: Message) -> Dict[str, Any]:
        """Инициирование нового исследования."""
        task_id = message.payload.get("task_id", message.id)
        
        # Создание контекста исследования
        hypothesis = message.payload.get("hypothesis", "")
        self.research_context["current_hypothesis"] = hypothesis
        
        # Параллельный запуск специализированных агентов
        tasks = [
            self._start_literature_review(hypothesis, task_id),
            self._start_data_collection(task_id),
            self._start_pattern_analysis(task_id)
        ]
        
        # Сохраняем информацию о задаче
        self.active_tasks[task_id] = {
            "status": "processing",
            "subtasks": ["literature", "data", "patterns"],
            "results": {},
            "start_time": self.core.get_current_time()
        }
        
        return {"task_id": task_id, "status": "research_initiated"}
    
    async def _start_literature_review(self, hypothesis: str, task_id: str):
        """Запуск агента для обзора литературы."""
        lit_message = Message(
            sender=self.agent_id,
            recipient="literature_agent",
            message_type="review_literature",
            payload={
                "hypothesis": hypothesis,
                "task_id": task_id,
                "context": self.research_context
            }
        )
        await self.core.route_message(lit_message)
    
    async def _start_data_collection(self, task_id: str):
        """Запуск агента для сбора данных."""
        data_message = Message(
            sender=self.agent_id,
            recipient="data_collection_agent",
            message_type="collect_chronobiotics_data",
            payload={
                "task_id": task_id,
                "parameters": self.research_context.get("parameters", {})
            }
        )
        await self.core.route_message(data_message)
    
    async def handle_data_result(self, message: Message):
        """Обработка результатов от агента данных."""
        task_id = message.payload.get("task_id")
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["results"]["data"] = message.payload
            self.active_tasks[task_id]["subtasks"].remove("data")
            
            # Проверяем, все ли подзадачи выполнены
            if not self.active_tasks[task_id]["subtasks"]:
                await self._synthesize_results(task_id)
    
    async def _synthesize_results(self, task_id: str):
        """Синтез результатов от всех агентов."""
        task = self.active_tasks[task_id]
        
        # Анализ и объединение результатов
        synthesis = {
            "hypothesis_evaluation": self._evaluate_hypothesis(task["results"]),
            "key_findings": self._extract_key_findings(task["results"]),
            "recommendations": self._generate_recommendations(task["results"]),
            "confidence_score": self._calculate_confidence(task["results"])
        }
        
        # Обновление контекста исследования
        self.research_context["findings"].append(synthesis)
        
        # Отправка финального результата
        response = Message(
            sender=self.agent_id,
            recipient=task["original_sender"],
            message_type="research_complete",
            payload={
                "task_id": task_id,
                "synthesis": synthesis,
                "research_context": self.research_context
            }
        )
        await self.core.route_message(response)
    
    def _evaluate_hypothesis(self, results: Dict) -> Dict:
        """Оценка гипотезы на основе всех полученных данных."""
        # Реализация бизнес-логики оценки
        return {"supported": True, "confidence": 0.85, "evidence": []}
    
    async def handle_generate_report(self, message: Message) -> Dict[str, Any]:
        """Генерация комплексного отчета."""
        report_type = message.payload.get("report_type", "full")
        
        report = {
            "metadata": {
                "generated_by": self.agent_id,
                "timestamp": self.core.get_current_time(),
                "report_type": report_type
            },
            "executive_summary": self._generate_executive_summary(),
            "methodology": self._describe_methodology(),
            "findings": self.research_context.get("findings", []),
            "conclusions": self._draw_conclusions(),
            "recommendations": self._generate_recommendations_for_report(),
            "appendices": {
                "literature_references": self.research_context.get("literature_references", []),
                "temporal_patterns": self.research_context.get("temporal_patterns", {})
            }
        }
        
        return {"report": report, "format": "json"}
