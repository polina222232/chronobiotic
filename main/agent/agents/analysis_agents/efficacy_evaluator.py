"""
Efficacy Evaluator Agent - evaluates chronobiotic efficacy
"""

from typing import Dict, Any, List

from ..base_agent import BaseAgentImplementation, AgentRole, AgentCapability
from ....core.agent_base import AgentTask, AgentResult


class EfficacyEvaluator(BaseAgentImplementation):
    """
    Agent for evaluating chronobiotic efficacy
    Assesses circadian rhythm modulation effects
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="EfficacyEvaluator",
            role=AgentRole.ANALYZER,
            config=config
        )
        self.add_capability(AgentCapability.PROPERTY_PREDICTION)
        
        # Efficacy scoring weights
        self.weights = {
            "phase_shift": 0.3,
            "amplitude_modulation": 0.2,
            "duration": 0.15,
            "specificity": 0.2,
            "safety_margin": 0.15
        }
    
    async def can_handle(self, task_type: str, input_data: Dict[str, Any]) -> bool:
        """Check if can handle efficacy evaluation"""
        return task_type in [
            "evaluate_efficacy",
            "compare_efficacy",
            "predict_effectiveness"
        ]
    
    async def process(self, task: AgentTask) -> AgentResult:
        """Process efficacy evaluation request"""
        task_type = task.type
        
        if task_type == "evaluate_efficacy":
            return await self._evaluate_efficacy(task.input_data)
        elif task_type == "compare_efficacy":
            return await self._compare_efficacy(task.input_data)
        else:
            return self._create_error_result(task.id, f"Unknown task type: {task_type}")
    
    async def _evaluate_efficacy(self, input_data: Dict) -> AgentResult:
        """Evaluate chronobiotic efficacy"""
        substance = input_data.get("substance", {})
        experimental_data = input_data.get("experimental_data", {})
        
        # Calculate efficacy scores
        phase_shift_score = self._evaluate_phase_shift(experimental_data)
        amplitude_score = self._evaluate_amplitude(experimental_data)
        duration_score = self._evaluate_duration(experimental_data)
        specificity_score = self._evaluate_specificity(substance, experimental_data)
        safety_score = self._evaluate_safety(experimental_data)
        
        # Calculate weighted score
        total_score = (
                phase_shift_score * self.weights["phase_shift"] +
                amplitude_score * self.weights["amplitude_modulation"] +
                duration_score * self.weights["duration"] +
                specificity_score * self.weights["specificity"] +
                safety_score * self.weights["safety_margin"]
        )
        
        # Determine efficacy category
        if total_score >= 0.8:
            category = "highly_efficacious"
        elif total_score >= 0.6:
            category = "efficacious"
        elif total_score >= 0.4:
            category = "moderately_efficacious"
        else:
            category = "low_efficacy"
        
        return AgentResult(
            task_id=f"efficacy_{hash(str(substance))}",
            success=True,
            data={
                "substance": substance.get("name", "Unknown"),
                "total_score": total_score,
                "category": category,
                "component_scores": {
                    "phase_shift": phase_shift_score,
                    "amplitude_modulation": amplitude_score,
                    "duration": duration_score,
                    "specificity": specificity_score,
                    "safety": safety_score
                },
                "recommendations": self._generate_recommendations(component_scores),
                "confidence": self._calculate_confidence(experimental_data)
            }
        )
    
    def _evaluate_phase_shift(self, data: Dict) -> float:
        """Evaluate phase shift efficacy"""
        phase_shift = data.get("phase_shift_hours", 0)
        target_shift = data.get("target_shift_hours", 2)
        
        if phase_shift >= target_shift:
            return 1.0
        else:
            return phase_shift / target_shift
    
    def _evaluate_amplitude(self, data: Dict) -> float:
        """Evaluate amplitude modulation"""
        amplitude_change = abs(data.get("amplitude_change_percent", 0))
        return min(amplitude_change / 50, 1.0)
    
    def _evaluate_duration(self, data: Dict) -> float:
        """Evaluate effect duration"""
        duration_hours = data.get("effect_duration_hours", 0)
        return min(duration_hours / 12, 1.0)
    
    def _evaluate_specificity(self, substance: Dict, data: Dict) -> float:
        """Evaluate target specificity"""
        return data.get("specificity_score", 0.5)
    
    def _evaluate_safety(self, data: Dict) -> float:
        """Evaluate safety margin"""
        safety_margin = data.get("safety_margin", 1)
        return min(safety_margin / 10, 1.0)
    
    def _generate_recommendations(self, scores: Dict) -> List[str]:
        """Generate recommendations based on scores"""
        recommendations = []
        
        if scores.get("phase_shift", 0) < 0.6:
            recommendations.append("Consider higher dose or different timing")
        if scores.get("amplitude_modulation", 0) < 0.5:
            recommendations.append("May need combination therapy for amplitude effects")
        
        return recommendations
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate confidence in evaluation"""
        return data.get("data_quality", 0.7)
    
    async def _compare_efficacy(self, input_data: Dict) -> AgentResult:
        """Compare efficacy of multiple substances"""
        substances = input_data.get("substances", [])
        
        evaluations = []
        for substance in substances:
            eval_result = await self._evaluate_efficacy({"substance": substance})
            evaluations.append(eval_result.data)
        
        # Sort by total_score
        evaluations.sort(key=lambda x: x["total_score"], reverse=True)
        
        return AgentResult(
            task_id=f"compare_{hash(str(substances))}",
            success=True,
            data={
                "comparisons": evaluations,
                "best_substance": evaluations[0]["substance"] if evaluations else None,
                "ranking": [e["substance"] for e in evaluations]
            }
        )
