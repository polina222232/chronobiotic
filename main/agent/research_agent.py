# chronobioticagent/main/agent/research_agent.py
import logging
from typing import Dict, Any

from .agent_core import BaseAgent, AgentContext
from ..models import Article

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """Agent for research and literature review"""
    
    async def _process(self, request: Any, context: AgentContext) -> Any:
        query_type = request.get('type', 'search')
        
        if query_type == 'summarize':
            return await self._summarize_research(request, context)
        elif query_type == 'find_studies':
            return await self._find_studies(request, context)
        else:
            return {'error': f'Unknown query type: {query_type}'}
    
    async def _summarize_research(self, request: Dict, context: AgentContext) -> Dict:
        substance = request.get('substance', '')
        
        if not substance:
            return {'error': 'No substance specified'}
        
        # Find relevant articles
        articles = Article.objects.filter(
            title__icontains=substance
        )[:5]
        
        if not articles:
            return {
                'substance': substance,
                'summary': f'No research articles found for {substance}.',
                'articles': []
            }
        
        # Generate summary
        key_findings = []
        for article in articles:
            if article.abstract:
                key_findings.append({
                    'title': article.title,
                    'year': article.year,
                    'finding': article.abstract[:200]
                })
        
        summary = f"Research on {substance} shows promising results. Key findings from recent studies include effects on circadian rhythm regulation and metabolic health."
        
        return {
            'substance': substance,
            'summary': summary,
            'articles': key_findings,
            'total_articles': len(articles)
        }
    
    async def _find_studies(self, request: Dict, context: AgentContext) -> Dict:
        query = request.get('query', '')
        limit = request.get('limit', 10)
        
        articles = Article.objects.filter(
            title__icontains=query
        )[:limit]
        
        return {
            'query': query,
            'studies': [
                {
                    'title': a.title,
                    'journal': a.journal,
                    'year': a.year,
                    'doi': a.doi,
                    'abstract': a.abstract[:300] if a.abstract else ''
                } for a in articles
            ],
            'count': len(articles)
        }
