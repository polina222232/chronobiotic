# chronobioticagent/main/agent/database_agent.py
import logging
from typing import Dict, Any

from django.db.models import Q

from .agent_core import BaseAgent, AgentContext
from ..models import Chronobiotic, Effect, Target, Article

logger = logging.getLogger(__name__)


class DatabaseAgent(BaseAgent):
    """Agent for database operations and search"""
    
    async def _on_initialize(self) -> bool:
        logger.info("DatabaseAgent initializing...")
        return True
    
    async def _process(self, request: Any, context: AgentContext) -> Any:
        query_type = request.get('type', 'search')
        
        if query_type == 'search':
            return await self._search(request, context)
        elif query_type == 'get_chronobiotic':
            return await self._get_chronobiotic(request, context)
        elif query_type == 'get_related':
            return await self._get_related(request, context)
        elif query_type == 'get_articles':
            return await self._get_articles(request, context)
        else:
            return {'error': f'Unknown query type: {query_type}'}
    
    async def _search(self, request: Dict, context: AgentContext) -> Dict:
        query = request.get('query', '')
        page = request.get('page', 1)
        per_page = request.get('per_page', 20)
        
        if not query:
            return {'error': 'No search query provided'}
        
        # Search in chronobiotics
        chronobiotics = Chronobiotic.objects.filter(
            Q(name__icontains=query) |
            Q(scientific_name__icontains=query) |
            Q(description__icontains=query) |
            Q(effect__icontains=query)
        )[:10]
        
        # Search in effects
        effects = Effect.objects.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query)
        )[:10]
        
        # Search in articles
        articles = Article.objects.filter(
            Q(title__icontains=query) |
            Q(abstract__icontains=query)
        )[:10]
        
        results = {
            'chronobiotics': [
                {
                    'id': c.id,
                    'name': c.name,
                    'description': c.description[:200],
                    'status': c.status
                } for c in chronobiotics
            ],
            'effects': [
                {
                    'id': e.id,
                    'name': e.name,
                    'chronobiotic': e.chronobiotic.name,
                    'description': e.description[:200]
                } for e in effects
            ],
            'articles': [
                {
                    'id': a.id,
                    'title': a.title,
                    'journal': a.journal,
                    'year': a.year,
                    'doi': a.doi
                } for a in articles
            ],
            'total_results': len(chronobiotics) + len(effects) + len(articles)
        }
        
        return results
    
    async def _get_chronobiotic(self, request: Dict, context: AgentContext) -> Dict:
        chronobiotic_id = request.get('id')
        name = request.get('name')
        
        if chronobiotic_id:
            try:
                chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
            except Chronobiotic.DoesNotExist:
                return {'error': 'Chronobiotic not found'}
        elif name:
            try:
                chronobiotic = Chronobiotic.objects.get(name__iexact=name)
            except Chronobiotic.DoesNotExist:
                return {'error': f'Chronobiotic "{name}" not found'}
        else:
            return {'error': 'Either id or name is required'}
        
        # Get related data
        effects = chronobiotic.effects.all()
        targets = ChronobioticTarget.objects.filter(chronobiotic=chronobiotic).select_related('target')
        articles = chronobiotic.articles.all()[:5]
        
        return {
            'id': chronobiotic.id,
            'name': chronobiotic.name,
            'scientific_name': chronobiotic.scientific_name,
            'description': chronobiotic.description,
            'mechanisms': chronobiotic.mechanisms,
            'effect': chronobiotic.effect,
            'source': chronobiotic.source,
            'dosage': chronobiotic.dosage,
            'safety': chronobiotic.safety,
            'molecular_formula': chronobiotic.molecular_formula,
            'molecular_weight': chronobiotic.molecular_weight,
            'smiles': chronobiotic.smiles,
            'effects': [
                {
                    'name': e.name,
                    'description': e.description,
                    'evidence_level': e.evidence_level
                } for e in effects
            ],
            'targets': [
                {
                    'name': t.target.name,
                    'full_name': t.target.full_name,
                    'affinity': t.affinity,
                    'mechanism': t.mechanism
                } for t in targets
            ],
            'recent_articles': [
                {
                    'title': a.title,
                    'journal': a.journal,
                    'year': a.year,
                    'doi': a.doi
                } for a in articles
            ]
        }
    
    async def _get_related(self, request: Dict, context: AgentContext) -> Dict:
        chronobiotic_id = request.get('id')
        limit = request.get('limit', 5)
        
        if not chronobiotic_id:
            return {'error': 'Chronobiotic id is required'}
        
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
        except Chronobiotic.DoesNotExist:
            return {'error': 'Chronobiotic not found'}
        
        # Find related chronobiotics through shared targets or effects
        target_ids = ChronobioticTarget.objects.filter(
            chronobiotic=chronobiotic
        ).values_list('target_id', flat=True)
        
        related_by_targets = ChronobioticTarget.objects.filter(
            target_id__in=target_ids
        ).exclude(chronobiotic=chronobiotic).values_list('chronobiotic_id', flat=True).distinct()[:limit]
        
        related = Chronobiotic.objects.filter(id__in=related_by_targets)
        
        return {
            'chronobiotic': chronobiotic.name,
            'related_chronobiotics': [
                {
                    'id': r.id,
                    'name': r.name,
                    'relationship': 'shared_targets'
                } for r in related
            ]
        }
    
    async def _get_articles(self, request: Dict, context: AgentContext) -> Dict:
        chronobiotic_id = request.get('chronobiotic_id')
        limit = request.get('limit', 10)
        
        if chronobiotic_id:
            try:
                chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
                articles = chronobiotic.articles.all()[:limit]
            except Chronobiotic.DoesNotExist:
                return {'error': 'Chronobiotic not found'}
        else:
            articles = Article.objects.all()[:limit]
        
        return {
            'articles': [
                {
                    'id': a.id,
                    'title': a.title,
                    'authors': a.authors[:200],
                    'journal': a.journal,
                    'year': a.year,
                    'doi': a.doi,
                    'abstract': a.abstract[:500] if a.abstract else ''
                } for a in articles
            ]
        }
