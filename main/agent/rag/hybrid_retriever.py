"""
Hybrid retriever that searches both local database and internet
"""

from typing import List, Dict, Any, Optional
from django.db import models
from django.db.models import Q
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result from any source"""
    content: str
    source: str  # 'database' or 'web'
    relevance_score: float
    metadata: Dict[str, Any]
    citation: Optional[str] = None


class HybridRetriever:
    """
    Hybrid retriever that searches:
    1. Local Chronobiotics database
    2. Web search (via external APIs)
    3. Knowledge Graph (KAG)
    """
    
    def __init__(self):
        self.db_priority = True  # Prioritize database results
        self.max_db_results = 10
        self.max_web_results = 5
    
    def search(self, query: str, use_web_fallback: bool = True) -> List[SearchResult]:
        """
        Search for information about query
        """
        results = []
        
        # Step 1: Search local database
        db_results = self._search_database(query)
        results.extend(db_results)
        
        # Step 2: If insufficient results, search web
        if use_web_fallback and len(db_results) < 3:
            logger.info(f"Insufficient DB results ({len(db_results)}), searching web...")
            web_results = self._search_web(query)
            results.extend(web_results)
        
        # Step 3: Also search Knowledge Graph if available
        kg_results = self._search_knowledge_graph(query)
        results.extend(kg_results)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def _search_database(self, query: str) -> List[SearchResult]:
        """
        Search Chronobiotics database models
        """
        results = []
        
        # Import models here to avoid circular imports
        from main.models import Chronobiotic, Bioclass, Mechanism, Effect, Targets, Articles, Synonyms
        
        # Search Chronobiotics by name, description, SMILES
        chronobiotics = Chronobiotic.objects.filter(
            Q(gname__icontains=query) |
            Q(description__icontains=query) |
            Q(smiles__icontains=query) |
            Q(iupacname__icontains=query)
        )[:self.max_db_results]
        
        for cb in chronobiotics:
            content = self._format_chronobiotic_content(cb)
            results.append(SearchResult(
                content=content,
                source='database',
                relevance_score=self._calculate_db_relevance(query, cb.gname, cb.description),
                metadata={
                    'id': cb.id,
                    'type': 'chronobiotic',
                    'name': cb.gname,
                    'smiles': cb.smiles,
                    'links': self._get_chronobiotic_links(cb)
                },
                citation=f"Chronobiotic database: {cb.gname}"
            ))
        
        # Search by Bioclass
        classes = Bioclass.objects.filter(nameclass__icontains=query)[:5]
        for bioclass in classes:
            related = Chronobiotic.objects.filter(classf=bioclass)[:5]
            content = f"Class: {bioclass.nameclass}\n"
            content += f"Related compounds: {', '.join([c.gname for c in related])}\n"
            content += f"Number of compounds: {Chronobiotic.objects.filter(classf=bioclass).count()}"
            results.append(SearchResult(
                content=content,
                source='database',
                relevance_score=0.8,
                metadata={'type': 'bioclass', 'name': bioclass.nameclass},
                citation=f"Bioclass: {bioclass.nameclass}"
            ))
        
        # Search by Mechanism
        mechanisms = Mechanism.objects.filter(mechanismname__icontains=query)[:5]
        for mechanism in mechanisms:
            related = Chronobiotic.objects.filter(mechanisms=mechanism)[:5]
            content = f"Mechanism: {mechanism.mechanismname}\n"
            content += f"Compounds with this mechanism: {', '.join([c.gname for c in related])}"
            results.append(SearchResult(
                content=content,
                source='database',
                relevance_score=0.75,
                metadata={'type': 'mechanism', 'name': mechanism.mechanismname},
                citation=f"Mechanism: {mechanism.mechanismname}"
            ))
        
        # Search by Effect
        effects = Effect.objects.filter(Effectname__icontains=query)[:5]
        for effect in effects:
            related = Chronobiotic.objects.filter(effect=effect)[:5]
            content = f"Effect: {effect.Effectname}\n"
            content += f"Compounds with this effect: {', '.join([c.gname for c in related])}"
            results.append(SearchResult(
                content=content,
                source='database',
                relevance_score=0.7,
                metadata={'type': 'effect', 'name': effect.Effectname},
                citation=f"Effect: {effect.Effectname}"
            ))
        
        # Search Targets
        targets = Targets.objects.filter(
            Q(targetsname__icontains=query) |
            Q(targetsfullname__icontains=query)
        )[:5]
        for target in targets:
            related = Chronobiotic.objects.filter(target=target)[:5]
            content = f"Target: {target.targetsname} - {target.targetsfullname}\n"
            content += f"Compounds targeting this: {', '.join([c.gname for c in related])}"
            results.append(SearchResult(
                content=content,
                source='database',
                relevance_score=0.7,
                metadata={'type': 'target', 'name': target.targetsname, 'fullname': target.targetsfullname},
                citation=f"Target: {target.targetsname}"
            ))
        
        # Search Synonyms
        synonyms = Synonyms.objects.filter(synonymsmname__icontains=query)[:5]
        for syn in synonyms:
            if syn.originalbiotic:
                content = f"Synonym: {syn.synonymsmname} → {syn.originalbiotic.gname}\n"
                content += f"Description: {syn.originalbiotic.description[:500]}"
                results.append(SearchResult(
                    content=content,
                    source='database',
                    relevance_score=0.85,
                    metadata={'type': 'synonym', 'name': syn.synonymsmname, 'original': syn.originalbiotic.gname},
                    citation=f"Synonym mapping: {syn.synonymsmname}"
                ))
        
        # Search Articles
        articles = Articles.objects.filter(articlename__icontains=query)[:5]
        for article in articles:
            related = Chronobiotic.objects.filter(articles=article)[:3]
            content = f"Article: {article.articlename}\n"
            content += f"URL: {article.articleurl}\n"
            content += f"Related compounds: {', '.join([c.gname for c in related])}"
            results.append(SearchResult(
                content=content,
                source='database',
                relevance_score=0.6,
                metadata={'type': 'article', 'title': article.articlename, 'url': article.articleurl},
                citation=article.articlename
            ))
        
        return results
    
    def _search_web(self, query: str) -> List[SearchResult]:
        """
        Search web for additional information
        """
        results = []
        
        try:
            # Import web search module
            from ..web.web_client import WebClient
            from ..web.scraper_engine import ScraperEngine
            
            web_client = WebClient()
            scraper = ScraperEngine()
            
            # Search PubMed/Google Scholar for scientific articles
            search_queries = [
                f"{query} chronobiotic",
                f"{query} circadian rhythm",
                f"{query} clinical trial",
                f"{query} mechanism of action"
            ]
            
            for sq in search_queries[:2]:  # Limit to 2 queries
                search_results = web_client.search(sq)
                for result in search_results[:3]:
                    # Scrape content if needed
                    content = self._extract_relevant_content(result, query)
                    if content:
                        results.append(SearchResult(
                            content=content,
                            source='web',
                            relevance_score=0.5,
                            metadata={
                                'url': result.get('url'),
                                'title': result.get('title'),
                                'snippet': result.get('snippet')
                            },
                            citation=result.get('title', 'Web source')
                        ))
        
        except Exception as e:
            logger.error(f"Web search failed: {e}")
        
        return results
    
    def _search_knowledge_graph(self, query: str) -> List[SearchResult]:
        """
        Search knowledge graph for relationships
        """
        results = []
        
        try:
            from ..kag.kag_service import KAGService
            
            kag = KAGService()
            kg_results = kag.query(query)
            
            for kg_result in kg_results[:3]:
                content = f"Knowledge Graph: {kg_result.get('content', '')}\n"
                content += f"Relationships: {kg_result.get('relationships', [])}"
                results.append(SearchResult(
                    content=content,
                    source='knowledge_graph',
                    relevance_score=kg_result.get('score', 0.6),
                    metadata=kg_result.get('metadata', {}),
                    citation=kg_result.get('source', 'Knowledge Graph')
                ))
        
        except Exception as e:
            logger.debug(f"KAG search not available: {e}")
        
        return results
    
    def _format_chronobiotic_content(self, chronobiotic) -> str:
        """Format chronobiotic object as readable text"""
        content = f"**{chronobiotic.gname}**\n\n"
        
        if chronobiotic.description:
            content += f"Description: {chronobiotic.description[:1000]}\n\n"
        
        if chronobiotic.smiles:
            content += f"SMILES: {chronobiotic.smiles}\n"
        
        if chronobiotic.iupacname:
            content += f"IUPAC Name: {chronobiotic.iupacname}\n"
        
        # Related information
        classes = list(chronobiotic.classf.all())
        if classes:
            content += f"Class: {', '.join([c.nameclass for c in classes])}\n"
        
        mechanisms = list(chronobiotic.mechanisms.all())
        if mechanisms:
            content += f"Mechanisms: {', '.join([m.mechanismname for m in mechanisms])}\n"
        
        effects = list(chronobiotic.effect.all())
        if effects:
            content += f"Effects: {', '.join([e.Effectname for e in effects])}\n"
        
        targets = list(chronobiotic.target.all())
        if targets:
            content += f"Targets: {', '.join([t.targetsname for t in targets])}\n"
        
        return content
    
    def _get_chronobiotic_links(self, chronobiotic) -> Dict[str, str]:
        """Get all external links for a chronobiotic"""
        links = {}
        if chronobiotic.pubchem:
            links['pubchem'] = chronobiotic.pubchem
        if chronobiotic.drugbank:
            links['drugbank'] = chronobiotic.drugbank
        if chronobiotic.chebi:
            links['chebi'] = chronobiotic.chebi
        if chronobiotic.kegg:
            links['kegg'] = chronobiotic.kegg
        return links
    
    def _calculate_db_relevance(self, query: str, name: str, description: str) -> float:
        """Calculate relevance score for database result"""
        query_lower = query.lower()
        score = 0.0
        
        # Exact name match - high relevance
        if query_lower == name.lower():
            score = 1.0
        # Name contains query
        elif query_lower in name.lower():
            score = 0.9
        # Description contains query
        elif description and query_lower in description.lower():
            score = 0.7
        else:
            score = 0.5
        
        return score
    
    def _extract_relevant_content(self, search_result: Dict, query: str) -> Optional[str]:
        """Extract relevant content from web search result"""
        snippet = search_result.get('snippet', '')
        title = search_result.get('title', '')
        
        if not snippet:
            return None
        
        content = f"**{title}**\n\n"
        content += f"Summary: {snippet}\n"
        content += f"Source: {search_result.get('url', '')}"
        
        return content
