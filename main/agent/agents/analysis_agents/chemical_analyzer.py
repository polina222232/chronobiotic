# chronobioticagent/main/agent/agents/analysis_agents/chemical_analyzer.py
"""
Chemical Analyzer Agent
Analyzes chemical properties, structures, and characteristics of chronobiotics
"""

import logging
from typing import Dict, Any, Optional, List

from ..base_agent import ChronobioticsBaseAgent, AgentContext

logger = logging.getLogger(__name__)


class ChemicalAnalyzerAgent(ChronobioticsBaseAgent):
    """
    Specialized agent for chemical analysis of chronobiotic substances
    Analyzes molecular structure, properties, and chemical characteristics
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.priority = self.priority.HIGH
        self._chemical_service = None
    
    async def _on_initialize(self) -> bool:
        """Initialize chemical analysis service"""
        try:
            # Initialize chemical service
            from ...chem.chemical_service import ChemicalService
            self._chemical_service = ChemicalService(self.config.get('chemical_config', {}))
            await self._chemical_service.initialize()
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ChemicalAnalyzerAgent: {e}")
            return False
    
    async def _process(self, request: Any, context: AgentContext) -> Any:
        """
        Process chemical analysis request

        Request formats:
        1. {"type": "analyze_structure", "smiles": "C[N+]1(C)...", "name": "melatonin"}
        2. {"type": "find_similar", "smiles": "...", "threshold": 0.8}
        3. {"type": "predict_properties", "substance": "melatonin"}
        4. {"type": "analyze_interaction", "substance1": "...", "substance2": "..."}
        """
        
        if isinstance(request, str):
            # Simple text query - try to extract chemical information
            return await self._handle_text_query(request, context)
        
        request_type = request.get('type', 'analyze_structure')
        
        if request_type == 'analyze_structure':
            return await self._analyze_structure(request, context)
        elif request_type == 'find_similar':
            return await self._find_similar_compounds(request, context)
        elif request_type == 'predict_properties':
            return await self._predict_properties(request, context)
        elif request_type == 'analyze_interaction':
            return await self._analyze_interaction(request, context)
        elif request_type == 'validate_smiles':
            return await self._validate_smiles(request, context)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    async def _handle_text_query(self, query: str, context: AgentContext) -> Dict[str, Any]:
        """Handle natural language chemical queries"""
        substance = self._detect_substance(query)
        
        if not substance:
            return {
                'error': 'Could not detect a specific substance in the query',
                'query': query,
                'suggestion': 'Please specify a substance name or SMILES string'
            }
        
        # Analyze the detected substance
        return await self._analyze_structure({'substance': substance, 'name': substance}, context)
    
    async def _analyze_structure(self, params: Dict, context: AgentContext) -> Dict[str, Any]:
        """Analyze molecular structure of a substance"""
        substance = params.get('substance') or params.get('smiles') or params.get('name')
        
        if not substance:
            return {'error': 'No substance, SMILES, or name provided'}
        
        result = {
            'substance': substance,
            'analysis_type': 'structure_analysis',
            'timestamp': None
        }
        
        try:
            from datetime import datetime
            result['timestamp'] = datetime.now().isoformat()
            
            # Get chemical information
            chem_info = await self._chemical_service.get_chemical_info(substance)
            
            if chem_info:
                result['molecular_formula'] = chem_info.get('formula')
                result['molecular_weight'] = chem_info.get('molecular_weight')
                result['smiles'] = chem_info.get('canonical_smiles')
                result['inchi'] = chem_info.get('inchi')
                result['inchikey'] = chem_info.get('inchikey')
                
                # Calculate additional properties
                properties = await self._chemical_service.calculate_properties(chem_info.get('canonical_smiles'))
                result['properties'] = properties
            
            # Add knowledge graph enrichment
            await self._enrich_with_kg([result], substance)
            
            # Add citations
            await self._add_citations(result, f"chemical analysis of {substance}")
            
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    async def _find_similar_compounds(self, params: Dict, context: AgentContext) -> Dict[str, Any]:
        """Find structurally similar compounds"""
        query_smiles = params.get('smiles') or params.get('substance')
        threshold = params.get('threshold', 0.7)
        limit = params.get('limit', 10)
        
        if not query_smiles:
            return {'error': 'No SMILES or substance provided for similarity search'}
        
        result = {
            'query': query_smiles,
            'threshold': threshold,
            'similar_compounds': [],
            'analysis_type': 'similarity_search'
        }
        
        try:
            similar = await self._chemical_service.find_similar(query_smiles, threshold, limit)
            
            for comp in similar:
                result['similar_compounds'].append({
                    'name': comp.get('name', 'Unknown'),
                    'smiles': comp.get('smiles'),
                    'similarity': comp.get('similarity_score', 0.0),
                    'properties': comp.get('properties', {})
                })
            
            result['count'] = len(result['similar_compounds'])
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    async def _predict_properties(self, params: Dict, context: AgentContext) -> Dict[str, Any]:
        """Predict chemical and biological properties"""
        substance = params.get('substance') or params.get('smiles')
        
        if not substance:
            return {'error': 'No substance provided for property prediction'}
        
        result = {
            'substance': substance,
            'predicted_properties': {},
            'analysis_type': 'property_prediction'
        }
        
        try:
            properties = await self._chemical_service.predict_properties(substance)
            
            result['predicted_properties'] = {
                'logP': properties.get('logP'),
                'solubility': properties.get('solubility'),
                'bioavailability': properties.get('bioavailability_score'),
                'toxicity_potential': properties.get('toxicity_potential'),
                'half_life': properties.get('half_life_hours'),
                'blood_brain_barrier': properties.get('bbb_permeable', False)
            }
            
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Property prediction failed: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    async def _analyze_interaction(self, params: Dict, context: AgentContext) -> Dict[str, Any]:
        """Analyze potential interactions between two substances"""
        substance1 = params.get('substance1')
        substance2 = params.get('substance2')
        
        if not substance1 or not substance2:
            return {'error': 'Both substance1 and substance2 are required'}
        
        result = {
            'substance1': substance1,
            'substance2': substance2,
            'interactions': [],
            'analysis_type': 'interaction_analysis'
        }
        
        try:
            interactions = await self._chemical_service.predict_interactions(substance1, substance2)
            
            for interaction in interactions:
                result['interactions'].append({
                    'type': interaction.get('type', 'unknown'),
                    'description': interaction.get('description'),
                    'severity': interaction.get('severity', 'unknown'),
                    'mechanism': interaction.get('mechanism'),
                    'confidence': interaction.get('confidence', 0.5)
                })
            
            result['has_interactions'] = len(result['interactions']) > 0
            result['risk_level'] = self._calculate_risk_level(result['interactions'])
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Interaction analysis failed: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    async def _validate_smiles(self, params: Dict, context: AgentContext) -> Dict[str, Any]:
        """Validate and canonicalize SMILES strings"""
        smiles = params.get('smiles')
        
        if not smiles:
            return {'error': 'No SMILES string provided'}
        
        result = {
            'input_smiles': smiles,
            'analysis_type': 'smiles_validation'
        }
        
        try:
            is_valid, canonical, error = await self._chemical_service.validate_smiles(smiles)
            
            result['is_valid'] = is_valid
            result['canonical_smiles'] = canonical
            result['error'] = error
            
            if is_valid:
                result['success'] = True
            else:
                result['success'] = False
                result['error_message'] = error
        
        except Exception as e:
            logger.error(f"SMILES validation failed: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def _calculate_risk_level(self, interactions: List[Dict]) -> str:
        """Calculate overall risk level from interactions"""
        if not interactions:
            return 'none'
        
        severities = [i.get('severity', 'low').lower() for i in interactions]
        
        if 'critical' in severities or 'high' in severities:
            return 'high'
        elif 'moderate' in severities:
            return 'moderate'
        elif 'low' in severities:
            return 'low'
        
        return 'unknown'
