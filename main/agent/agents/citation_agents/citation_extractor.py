# chronobioticagent/main/agent/agents/citation_agents/citation_extractor.py
"""
Citation Extractor Agent - extracts and manages citations
"""

import re
from typing import Dict, Any, List

from ..base_agent import BaseAgentImplementation, AgentRole, AgentCapability
from ....core.agent_base import AgentTask, AgentResult


class CitationExtractor(BaseAgentImplementation):
    """
    Agent for extracting and managing citations
    Handles citation detection, extraction, and formatting
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="CitationExtractor",
            role=AgentRole.ASSISTANT,
            config=config
        )
        self.add_capability(AgentCapability.CITATION_MANAGEMENT)
        
        # Citation patterns
        self.citation_patterns = {
            "doi": r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
            "pmid": r"PMID:\s*(\d+)",
            "bibtex": r"@\w+\{([^,]+),",
            "inline": r"\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4})\)"
        }
    
    async def can_handle(self, task_type: str, input_data: Dict[str, Any]) -> bool:
        """Check if can handle citation tasks"""
        return task_type in [
            "extract_citations",
            "format_citations",
            "validate_citation",
            "generate_bibliography"
        ]
    
    async def process(self, task: AgentTask) -> AgentResult:
        """Process citation request"""
        task_type = task.type
        
        if task_type == "extract_citations":
            return await self._extract_citations(task.input_data)
        elif task_type == "format_citations":
            return await self._format_citations(task.input_data)
        elif task_type == "generate_bibliography":
            return await self._generate_bibliography(task.input_data)
        else:
            return self._create_error_result(task.id, f"Unknown task type: {task_type}")
    
    async def _extract_citations(self, input_data: Dict) -> AgentResult:
        """Extract citations from text"""
        text = input_data.get("text", "")
        
        citations = []
        
        # Extract DOIs
        dois = re.findall(self.citation_patterns["doi"], text, re.IGNORECASE)
        for doi in dois:
            citations.append({
                "type": "doi",
                "identifier": doi,
                "text": f"DOI: {doi}"
            })
        
        # Extract PMIDs
        pmids = re.findall(self.citation_patterns["pmid"], text)
        for pmid in pmids:
            citations.append({
                "type": "pmid",
                "identifier": pmid,
                "text": f"PMID: {pmid}"
            })
        
        # Extract inline citations
        inline = re.findall(self.citation_patterns["inline"], text)
        for citation in inline:
            citations.append({
                "type": "inline",
                "text": citation,
                "format": "author-year"
            })
        
        return AgentResult(
            task_id=f"extract_{hash(text[:100])}",
            success=True,
            data={
                "citations_found": len(citations),
                "citations": citations,
                "summary": self._summarize_citations(citations)
            }
        )
    
    async def _format_citations(self, input_data: Dict) -> AgentResult:
        """Format citations in specified style"""
        citations = input_data.get("citations", [])
        style = input_data.get("style", "apa")  # apa, mla, chicago, bibtex
        
        formatted = []
        for citation in citations:
            if style == "apa":
                formatted.append(self._format_apa(citation))
            elif style == "mla":
                formatted.append(self._format_mla(citation))
            elif style == "chicago":
                formatted.append(self._format_chicago(citation))
            elif style == "bibtex":
                formatted.append(self._format_bibtex(citation))
            else:
                formatted.append(str(citation))
        
        return AgentResult(
            task_id=f"format_{hash(str(citations))}",
            success=True,
            data={
                "style": style,
                "formatted_citations": formatted,
                "bibliography": "\n\n".join(formatted)
            }
        )
    
    def _format_apa(self, citation: Dict) -> str:
        """Format citation in APA style"""
        authors = citation.get("authors", ["Unknown"])
        year = citation.get("year", "n.d.")
        title = citation.get("title", "Untitled")
        journal = citation.get("journal", "")
        
        author_str = ", ".join(authors[:-1]) + (" & " + authors[-1] if len(authors) > 1 else authors[0])
        
        return f"{author_str} ({year}). {title}. {journal}."
    
    def _format_mla(self, citation: Dict) -> str:
        """Format citation in MLA style"""
        authors = citation.get("authors", ["Unknown"])
        title = citation.get("title", "Untitled")
        journal = citation.get("journal", "")
        year = citation.get("year", "n.d.")
        
        author_str = ", ".join(authors)
        
        return f"{author_str}. \"{title}.\" {journal}, {year}."
    
    def _format_chicago(self, citation: Dict) -> str:
        """Format citation in Chicago style"""
        authors = citation.get("authors", ["Unknown"])
        year = citation.get("year", "n.d.")
        title = citation.get("title", "Untitled")
        journal = citation.get("journal", "")
        
        author_str = " and ".join(authors)
        
        return f"{author_str}. {year}. \"{title}.\" {journal}."
    
    def _format_bibtex(self, citation: Dict) -> str:
        """Format citation as BibTeX entry"""
        citation_id = citation.get("id", "ref1")
        authors = citation.get("authors", [])
        year = citation.get("year", "n.d.")
        title = citation.get("title", "Untitled")
        journal = citation.get("journal", "")
        
        author_str = " and ".join(authors)
        
        return f"""@article{{{citation_id},
    author = {{{author_str}}},
    title = {{{title}}},
    journal = {{{journal}}},
    year = {{{year}}}
}}"""
    
    async def _generate_bibliography(self, input_data: Dict) -> AgentResult:
        """Generate complete bibliography from citations"""
        citations = input_data.get("citations", [])
        style = input_data.get("style", "apa")
        
        # Remove duplicates
        unique_citations = self._deduplicate_citations(citations)
        
        # Format all citations
        formatted = await self._format_citations({
            "citations": unique_citations,
            "style": style
        })
        
        return AgentResult(
            task_id=f"bib_{hash(str(citations))}",
            success=True,
            data={
                "style": style,
                "citation_count": len(unique_citations),
                "bibliography": formatted.data["bibliography"],
                "references": formatted.data["formatted_citations"]
            }
        )
    
    def _deduplicate_citations(self, citations: List[Dict]) -> List[Dict]:
        """Remove duplicate citations"""
        seen = set()
        unique = []
        
        for citation in citations:
            key = f"{citation.get('doi')}{citation.get('pmid')}{citation.get('title')}"
            if key not in seen:
                seen.add(key)
                unique.append(citation)
        
        return unique
    
    def _summarize_citations(self, citations: List[Dict]) -> str:
        """Generate summary of extracted citations"""
        if not citations:
            return "No citations found."
        
        types = {}
        for citation in citations:
            t = citation.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        
        summary = f"Found {len(citations)} citations: "
        summary += ", ".join([f"{count} {t}" for t, count in types.items()])
        return summary
