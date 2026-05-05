"""
Main Chronobiotics Agent
Orchestrates all other agents for comprehensive chronobiotics research
"""
import logging
from typing import Dict, Any

from .agent_core import BaseAgent, AgentContext
from .agent_manager import AgentManager
from .database_agent import DatabaseAgent
from .research_agent import ResearchAgent

logger = logging.getLogger(__name__)


class ChronobioticsAgent(BaseAgent):
    """Main orchestrator agent for chronobiotics research"""
    
    async def _on_initialize(self) -> bool:
        logger.info("ChronobioticsAgent initializing...")
        
        # Register and create sub-agents
        manager = AgentManager()
        
        # Register agent types
        manager.register_agent_type('database_agent', DatabaseAgent)
        manager.register_agent_type('research_agent', ResearchAgent)
        
        # Create agents
        await manager.create_agent('database_agent', 'database_agent', {})
        await manager.create_agent('research_agent', 'research_agent', {})
        
        logger.info("ChronobioticsAgent initialized successfully")
        return True
    
    async def _process(self, request: Any, context: AgentContext) -> Any:
        """Process user requests"""
        
        if isinstance(request, str):
            return await self._handle_query(request, context)
        
        query_type = request.get('type', 'chat')
        
        if query_type == 'search':
            return await self._search(request, context)
        elif query_type == 'analyze':
            return await self._analyze(request, context)
        elif query_type == 'ask':
            return await self._ask(request, context)
        else:
            return await self._chat(request.get('message', ''), context)
    
    async def _handle_query(self, query: str, context: AgentContext) -> Dict:
        """Handle natural language query"""
        query_lower = query.lower()
        
        # Detect intent
        if any(word in query_lower for word in ['what is', 'tell me about', 'describe']):
            # Extract substance name
            for word in ['melatonin', 'resveratrol', 'curcumin', 'berberine', 'theanine']:
                if word in query_lower:
                    return await self._get_chronobiotic_info(word, context)
        
        # Default to search
        return await self._search({'query': query}, context)
    
    async def _search(self, request: Dict, context: AgentContext) -> Dict:
        """Search the database"""
        manager = AgentManager()
        response = await manager.send_request('database_agent', {
            'type': 'search',
            'query': request.get('query', '')
        }, context)
        
        if response.success:
            return response.data
        return {'error': response.error}
    
    async def _get_chronobiotic_info(self, name: str, context: AgentContext) -> Dict:
        """Get detailed information about a chronobiotic"""
        manager = AgentManager()
        response = await manager.send_request('database_agent', {
            'type': 'get_chronobiotic',
            'name': name
        }, context)
        
        if response.success:
            return response.data
        return {'error': response.error}
    
    async def _analyze(self, request: Dict, context: AgentContext) -> Dict:
        """Analyze a chronobiotic"""
        substance = request.get('substance')
        
        if not substance:
            return {'error': 'No substance specified for analysis'}
        
        manager = AgentManager()
        
        # Get basic info
        info_response = await manager.send_request('database_agent', {
            'type': 'get_chronobiotic',
            'name': substance
        }, context)
        
        # Get related research
        research_response = await manager.send_request('research_agent', {
            'type': 'summarize',
            'substance': substance
        }, context)
        
        return {
            'substance': substance,
            'information': info_response.data if info_response.success else None,
            'research_summary': research_response.data if research_response.success else None,
            'analysis_complete': True
        }
    
    async def _ask(self, request: Dict, context: AgentContext) -> Dict:
        """Answer a specific question"""
        question = request.get('question', '')
        
        # Simple Q&A based on intent
        question_lower = question.lower()
        
        if 'mechanism' in question_lower:
            return await self._explain_mechanism(question_lower, context)
        elif 'benefit' in question_lower or 'effect' in question_lower:
            return await self._explain_benefits(question_lower, context)
        elif 'dosage' in question_lower:
            return await self._explain_dosage(question_lower, context)
        else:
            return await self._get_chronobiotic_info(self._extract_substance(question_lower), context)
    
    async def _chat(self, message: str, context: AgentContext) -> Dict:
        """General chat response"""
        # Simple response generation
        return {
            'response': self._generate_response(message),
            'type': 'chat'
        }
    
    def _extract_substance(self, text: str) -> str:
        substances = ['melatonin', 'resveratrol', 'curcumin', 'berberine', 'theanine']
        for substance in substances:
            if substance in text:
                return substance
        return 'chronobiotics'
    
    async def _explain_mechanism(self, text: str, context: AgentContext) -> Dict:
        substance = self._extract_substance(text)
        
        mechanisms = {
            'melatonin': 'Melatonin binds to MT1 and MT2 receptors in the suprachiasmatic nucleus, regulating the sleep-wake cycle.',
            'resveratrol': 'Resveratrol activates SIRT1 and AMPK pathways, influencing circadian gene expression.',
            'curcumin': 'Curcumin modulates NF-κB and circadian clock genes (Clock, Bmal1).',
            'berberine': 'Berberine activates AMPK, influencing circadian metabolism.',
            'theanine': 'L-Theanine increases GABA, serotonin, and dopamine levels, promoting relaxation.'
        }
        
        return {
            'substance': substance,
            'mechanism': mechanisms.get(substance, 'Mechanism information not available'),
            'type': 'mechanism'
        }
    
    async def _explain_benefits(self, text: str, context: AgentContext) -> Dict:
        substance = self._extract_substance(text)
        
        benefits = {
            'melatonin': 'Improves sleep quality, reduces jet lag, supports circadian rhythm regulation.',
            'resveratrol': 'Antioxidant, anti-aging, cardiovascular support, circadian modulation.',
            'curcumin': 'Anti-inflammatory, antioxidant, supports joint and brain health.',
            'berberine': 'Blood sugar regulation, lipid management, gut health.',
            'theanine': 'Reduces stress, improves sleep quality, enhances focus.'
        }
        
        return {
            'substance': substance,
            'benefits': benefits.get(substance, 'Benefit information not available'),
            'type': 'benefits'
        }
    
    async def _explain_dosage(self, text: str, context: AgentContext) -> Dict:
        substance = self._extract_substance(text)
        
        dosages = {
            'melatonin': 'Typically 0.5-5mg before bedtime.',
            'resveratrol': 'Typically 150-500mg daily.',
            'curcumin': 'Typically 500-2000mg daily with black pepper for absorption.',
            'berberine': 'Typically 500mg 2-3 times daily.',
            'theanine': 'Typically 100-400mg daily.'
        }
        
        return {
            'substance': substance,
            'dosage': dosages.get(substance, 'Dosage information not available'),
            'type': 'dosage'
        }
    
    def _generate_response(self, message: str) -> str:
        """Generate a friendly response"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm your Chronobiotics Research Assistant. How can I help you today?"
        
        if 'thank' in message_lower:
            return "You're welcome! Feel free to ask me anything about chronobiotics."
        
        if 'help' in message_lower:
            return """I can help you with:
- 🔬 **Compound analysis** - Learn about melatonin, resveratrol, curcumin, and more
- ⚙️ **Mechanisms** - Understand how chronobiotics affect circadian rhythms
- 📊 **Comparisons** - Compare different compounds
- 📚 **Research** - Find recent studies and clinical evidence
- 💊 **Applications** - Dosage, timing, and practical use

What would you like to know?"""
        
        return f"I understand you're asking about: {message[:100]}\n\nHow can I assist you with chronobiotics research?"
