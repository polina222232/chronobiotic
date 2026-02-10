# agent_monitor.py
import asyncio
import time
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import statistics
import json
from prometheus_client import Counter, Gauge, Histogram, start_http_server


@dataclass
class AlertRule:
    metric: str
    threshold: float
    condition: str  # "gt", "lt", "eq"
    severity: str  # "warning", "error", "critical"
    cooldown: int = 300  # seconds


class AgentMonitor:
    """Комплексная система мониторинга и алертинга."""
    
    def __init__(self, agent_manager: 'AgentManager', prometheus_port: int = 9090):
        self.manager = agent_manager
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Dict] = {}
        self.anomaly_detector = AnomalyDetector()
        
        # Prometheus метрики
        self.agent_status_gauge = Gauge('agent_status', 'Agent status', ['agent_id', 'agent_type'])
        self.message_rate_counter = Counter('messages_total', 'Total messages', ['message_type'])
        self.response_time_histogram = Histogram('response_time_seconds', 'Response time histogram')
        
        # Запуск Prometheus сервера
        start_http_server(prometheus_port)
    
    async def start_monitoring(self):
        """Запуск сбора метрик."""
        while True:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await self._detect_anomalies()
                await asyncio.sleep(5)  # Интервал сбора
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    async def _collect_metrics(self):
        """Сбор метрик со всех агентов."""
        current_time = time.time()
        
        for agent_id, agent in self.manager.agents.items():
            # Базовые метрики
            metrics = {
                "timestamp": current_time,
                "status": agent.status.value,
                "queue_size": agent.message_queue.qsize() if hasattr(agent, 'message_queue') else 0,
                "memory_usage": self._get_agent_memory(agent_id),
                "cpu_usage": self._get_agent_cpu(agent_id)
            }
            
            # Сохранение в историю
            if agent_id not in self.metrics_history:
                self.metrics_history[agent_id] = []
            self.metrics_history[agent_id].append(metrics)
            
            # Ограничение истории
            if len(self.metrics_history[agent_id]) > 1000:
                self.metrics_history[agent_id] = self.metrics_history[agent_id][-1000:]
            
            # Обновление Prometheus
            self.agent_status_gauge.labels(
                agent_id=agent_id,
                agent_type=agent.agent_type.value
            ).set(self._status_to_numeric(agent.status))
    
    def _status_to_numeric(self, status) -> int:
        """Конвертация статуса в числовое значение для метрик."""
        status_map = {
            "created": 0,
            "idle": 1,
            "processing": 2,
            "error": 3,
            "terminated": 4
        }
        return status_map.get(status.value, 0)
    
    async def _check_alerts(self):
        """Проверка условий алертов."""
        for rule in self.alert_rules:
            for agent_id, history in self.metrics_history.items():
                if history:
                    latest = history[-1]
                    metric_value = latest.get(rule.metric, 0)
                    
                    if self._evaluate_condition(metric_value, rule):
                        await self._trigger_alert(agent_id, rule, metric_value)
    
    def _evaluate_condition(self, value: float, rule: AlertRule) -> bool:
        """Оценка условия алерта."""
        conditions = {
            "gt": lambda x: x > rule.threshold,
            "lt": lambda x: x < rule.threshold,
            "eq": lambda x: x == rule.threshold,
        }
        return conditions.get(rule.condition, lambda x: False)(value)
    
    async def _trigger_alert(self, agent_id: str, rule: AlertRule, value: float):
        """Активация алерта."""
        alert_key = f"{agent_id}_{rule.metric}"
        
        # Проверка cooldown периода
        if alert_key in self.active_alerts:
            last_trigger = self.active_alerts[alert_key]["last_trigger"]
            if time.time() - last_trigger < rule.cooldown:
                return
        
        # Создание алерта
        alert = {
            "agent_id": agent_id,
            "metric": rule.metric,
            "value": value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "severity": rule.severity,
            "timestamp": time.time(),
            "last_trigger": time.time()
        }
        
        self.active_alerts[alert_key] = alert
        
        # Отправка уведомления
        await self._send_notification(alert)
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Генерация данных для веб-дашборда."""
        dashboard = {
            "system_health": self.manager.get_system_health(),
            "active_alerts": list(self.active_alerts.values()),
            "performance_metrics": self._calculate_performance_metrics(),
            "agent_details": self._get_agent_details(),
            "resource_utilization": self._get_resource_utilization()
        }
        return dashboard
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Расчет ключевых метрик производительности."""
        metrics = {
            "avg_response_time": 0.0,
            "success_rate": 1.0,
            "throughput": 0.0
        }
        
        # Анализ истории сообщений для расчета метрик
        # ... реализация анализа ...
        
        return metrics


class AnomalyDetector:
    """Детектор аномалий на основе машинного обучения."""
    
    def __init__(self):
        self.baselines: Dict[str, Dict] = {}
    
    def train_baseline(self, agent_id: str, metrics_history: List[Dict]):
        """Обучение базовой модели нормального поведения."""
        if len(metrics_history) < 100:
            return
        
        # Расчет статистических характеристик
        values = [m["queue_size"] for m in metrics_history]
        
        self.baselines[agent_id] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "percentiles": {
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)]
            }
        }
    
    def detect_anomaly(self, agent_id: str, current_metrics: Dict) -> bool:
        """Обнаружение аномалий."""
        if agent_id not in self.baselines:
            return False
        
        baseline = self.baselines[agent_id]
        queue_size = current_metrics.get("queue_size", 0)
        
        # Простое правило: 3 сигмы от среднего
        if queue_size > baseline["mean"] + 3 * baseline["std"]:
            return True
        
        # Или превышение 99 персентиля
        if queue_size > baseline["percentiles"]["p99"]:
            return True
        
        return False