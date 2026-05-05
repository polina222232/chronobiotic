from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from main.models import Chronobiotic
from ..base_agent import BaseAgent, AgentContext, AgentResult


class SubstanceAnalyzerAgent(BaseAgent):
    """
    Агент для анализа веществ-хронобиотиков
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("substance_analyzer", config)
        self.pubchem_api = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def get_capabilities(self) -> List[str]:
        return ["chemical_analysis", "substance_research", "property_prediction"]
    
    async def validate_input(self, input_data: Any) -> bool:
        """Валидация входных данных"""
        if isinstance(input_data, str):
            return bool(input_data.strip())
        elif isinstance(input_data, dict):
            return bool(input_data.get("substance_name") or input_data.get("smiles"))
        return False
    
    async def process(self,
                      input_data: Any,
                      context: AgentContext) -> AgentResult:
        """
        Анализ вещества-хронобиотика
        """
        await self.before_process(input_data, context)
        start_time = datetime.now()
        
        try:
            # Извлекаем информацию о веществе
            substance_info = await self._extract_substance_info(input_data)
            
            # Ищем в базе данных
            db_result = await self._search_database(substance_info)
            
            # Получаем данные из внешних API
            external_data = await self._fetch_external_data(substance_info)
            
            # Анализируем свойства
            properties = await self._analyze_properties(db_result, external_data)
            
            # Генерируем отчет
            report = self._generate_report(substance_info, db_result, properties)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                success=True,
                data={
                    "substance": substance_info,
                    "database_info": db_result,
                    "external_data": external_data,
                    "properties": properties,
                    "report": report
                },
                citations=self._extract_citations(db_result, external_data),
                confidence=self._calculate_confidence(db_result, external_data),
                processing_time=processing_time,
                metadata={
                    "source": "substance_analyzer",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            await self.on_error(e, context)
            return AgentResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        finally:
            await self.after_process(AgentResult(success=True))
    
    async def _extract_substance_info(self, input_data: Any) -> Dict:
        """Извлечение информации о веществе"""
        if isinstance(input_data, str):
            return {"name": input_data, "query_type": "name"}
        elif isinstance(input_data, dict):
            return input_data
        return {"raw": str(input_data)}
    
    async def _search_database(self, substance_info: Dict) -> Optional[Dict]:
        """Поиск в базе данных Django"""
        try:
            query = substance_info.get("name", "")
            if query:
                chronobiotic = await Chronobiotic.objects.filter(
                    gname__icontains=query
                ).afirst()
                
                if chronobiotic:
                    return {
                        "found": True,
                        "name": chronobiotic.gname,
                        "smiles": chronobiotic.smiles,
                        "description": chronobiotic.description,
                        "mechanisms": list(chronobiotic.mechanisms.all().values()),
                        "targets": list(chronobiotic.target.all().values()),
                        "effects": list(chronobiotic.effect.all().values())
                    }
        except Exception as e:
            self.logger.warning(f"Database search failed: {e}")
        
        return {"found": False}
    
    async def _fetch_external_data(self, substance_info: Dict) -> Dict:
        """Получение данных из внешних API"""
        external_data = {}
        
        # Получаем данные из PubChem
        if "smiles" in substance_info:
            pubchem_data = await self._query_pubchem(substance_info["smiles"])
            if pubchem_data:
                external_data["pubchem"] = pubchem_data
        
        return external_data
    
    async def _query_pubchem(self, smiles: str) -> Optional[Dict]:
        """Запрос к PubChem API"""
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.pubchem_api}/compound/smiles/{smiles}/property/MolecularWeight,CanonicalSMILES/JSON"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception as e:
                self.logger.error(f"PubChem query failed: {e}")
        return None
    
    async def _analyze_properties(self, db_result: Dict, external_data: Dict) -> Dict:
        """Анализ свойств вещества"""
        properties = {
            "chemical_properties": {},
            "biological_properties": {},
            "safety_profile": {}
        }
        
        # Анализ из базы данных
        if db_result.get("found"):
            properties["biological_properties"]["mechanisms"] = db_result.get("mechanisms", [])
            properties["biological_properties"]["targets"] = db_result.get("targets", [])
            properties["biological_properties"]["effects"] = db_result.get("effects", [])
        
        # Анализ из PubChem
        if external_data.get("pubchem"):
            pc_data = external_data["pubchem"]
            if "PropertyTable" in pc_data:
                props = pc_data["PropertyTable"]["Properties"][0]
                properties["chemical_properties"]["molecular_weight"] = props.get("MolecularWeight")
                properties["chemical_properties"]["canonical_smiles"] = props.get("CanonicalSMILES")
        
        return properties
    
    def _generate_report(self, substance_info: Dict, db_result: Dict, properties: Dict) -> str:
        """Генерация отчета"""
        report_parts = []
        
        report_parts.append(f"# Анализ вещества: {substance_info.get('name', 'Unknown')}\n")
        
        if db_result.get("found"):
            report_parts.append("## Информация из базы данных")
            report_parts.append(f"- **Название**: {db_result['name']}")
            report_parts.append(f"- **SMILES**: {db_result['smiles']}")
            report_parts.append(f"- **Описание**: {db_result['description'][:200]}...")
            
            if properties["biological_properties"].get("effects"):
                report_parts.append("\n## Эффекты")
                for effect in properties["biological_properties"]["effects"]:
                    report_parts.append(f"- {effect.get('Effectname', 'Unknown')}")
        else:
            report_parts.append("## Информация в базе данных не найдена")
        
        report_parts.append("\n## Химические свойства")
        for key, value in properties["chemical_properties"].items():
            if value:
                report_parts.append(f"- **{key}**: {value}")
        
        return "\n".join(report_parts)
    
    def _extract_citations(self, db_result: Dict, external_data: Dict) -> List[Dict]:
        """Извлечение цитирований"""
        citations = []
        
        if db_result.get("found"):
            citations.append({
                "source": "Chronobiotics Database",
                "type": "internal",
                "relevance": 1.0
            })
        
        if external_data.get("pubchem"):
            citations.append({
                "source": "PubChem",
                "type": "external_api",
                "url": "https://pubchem.ncbi.nlm.nih.gov",
                "relevance": 0.9
            })
        
        return citations
    
    def _calculate_confidence(self, db_result: Dict, external_data: Dict) -> float:
        """Расчет уверенности в результате"""
        confidence = 0.5  # базовое значение
        
        if db_result.get("found"):
            confidence += 0.3
        
        if external_data.get("pubchem"):
            confidence += 0.2
        
        return min(confidence, 1.0)
