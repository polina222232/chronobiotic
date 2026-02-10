"""
Chemical Models Interface for Agent System.

Provides specialized methods for chemical data operations.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from django.db.models import Q

from main.models import Chronobiotic
from .db_manager import get_db_manager

logger = logging.getLogger(__name__)


class ChemicalModelManager:
    """Manages chemical data operations for the agent system."""

    def __init__(self):
        self.db = get_db_manager()

    def extract_chemical_properties(self, chronobiotic_id: int) -> Dict[str, Any]:
        """
        Extract chemical properties from chronobiotic data.

        Args:
            chronobiotic_id: ID of the chronobiotic

        Returns:
            Dictionary of chemical properties
        """
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)

            properties = {
                "basic": {
                    "generic_name": chronobiotic.gname,
                    "smiles": chronobiotic.smiles,
                    "molecular_formula": chronobiotic.molecula,
                    "iupac_name": chronobiotic.iupacname,
                },
                "external_links": {
                    "pubchem": chronobiotic.pubchem,
                    "chemspider": chronobiotic.chemspider,
                    "drugbank": chronobiotic.drugbank,
                    "chebi": chronobiotic.chebi,
                    "uniprot": chronobiotic.uniprot,
                    "kegg": chronobiotic.kegg,
                    "selleckchem": chronobiotic.selleckchem,
                },
                "estimated_properties": self._estimate_properties_from_smiles(
                    chronobiotic.smiles
                ),
            }

            return properties

        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return {}
        except Exception as e:
            logger.error(f"Error extracting chemical properties: {e}")
            return {}

    def _estimate_properties_from_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Estimate chemical properties from SMILES string.
        This is a simplified version - in production, integrate with RDKit.

        Args:
            smiles: SMILES string

        Returns:
            Estimated properties
        """
        properties = {}

        try:
            # Simple pattern matching for basic properties
            # In production, use RDKit for accurate calculations

            # Count atoms (simplified)
            atom_counts = {}
            for char in smiles:
                if char.isupper():
                    # Simple atom detection (not accurate for all cases)
                    atom = char
                    if char in atom_counts:
                        atom_counts[atom] += 1
                    else:
                        atom_counts[atom] = 1

            properties["atom_counts"] = atom_counts

            # Estimate molecular weight (very rough approximation)
            atomic_weights = {
                "C": 12.01,
                "H": 1.008,
                "O": 16.00,
                "N": 14.01,
                "S": 32.06,
                "P": 30.97,
                "F": 19.00,
                "Cl": 35.45,
                "Br": 79.90,
                "I": 126.90,
            }

            estimated_weight = 0
            for atom, count in atom_counts.items():
                if atom in atomic_weights:
                    estimated_weight += atomic_weights[atom] * count

            properties["estimated_molecular_weight"] = round(estimated_weight, 2)

            # Detect rings (rough estimation)
            ring_count = (
                smiles.count("1")
                + smiles.count("2")
                + smiles.count("3")
                + smiles.count("4")
                + smiles.count("5")
                + smiles.count("6")
            )
            properties["ring_count"] = ring_count

            # Detect functional groups (simplified)
            functional_groups = {
                "hydroxyl": smiles.count("O") > 0,
                "amine": smiles.count("N") > 0,
                "carboxyl": "C(=O)O" in smiles,
                "amide": "C(=O)N" in smiles,
                "ester": "C(=O)O" in smiles and "O" in smiles.split("C(=O)O")[1],
                "ether": "O" in smiles
                and "C-O-C" in smiles.replace("(", "").replace(")", ""),
                "halogen": any(h in smiles for h in ["F", "Cl", "Br", "I"]),
            }
            properties["functional_groups"] = functional_groups

            # Estimate solubility (very rough)
            hydrophilic_atoms = atom_counts.get("O", 0) + atom_counts.get("N", 0)
            hydrophobic_atoms = atom_counts.get("C", 0) + atom_counts.get("H", 0)

            if hydrophobic_atoms > 0:
                hl_ratio = hydrophilic_atoms / hydrophobic_atoms
                if hl_ratio > 0.5:
                    solubility = "highly soluble"
                elif hl_ratio > 0.2:
                    solubility = "moderately soluble"
                else:
                    solubility = "poorly soluble"
            else:
                solubility = "unknown"

            properties["estimated_solubility"] = solubility

            # Lipinski's Rule of Five compliance (simplified)
            molecular_weight_ok = estimated_weight <= 500
            h_bond_donors = smiles.count("N") + smiles.count("O[H]")  # Simplified
            h_bond_acceptors = smiles.count("O") + smiles.count("N")  # Simplified
            logp_estimated = len(smiles) / 10  # Very rough approximation

            lipinski_compliant = (
                molecular_weight_ok
                and h_bond_donors <= 5
                and h_bond_acceptors <= 10
                and logp_estimated <= 5
            )

            properties["lipinski_rule_of_five"] = {
                "molecular_weight_ok": molecular_weight_ok,
                "h_bond_donors_ok": h_bond_donors <= 5,
                "h_bond_acceptors_ok": h_bond_acceptors <= 10,
                "logp_estimated_ok": logp_estimated <= 5,
                "is_compliant": lipinski_compliant,
                "h_bond_donors": h_bond_donors,
                "h_bond_acceptors": h_bond_acceptors,
                "estimated_logp": round(logp_estimated, 2),
            }

            return properties

        except Exception as e:
            logger.error(f"Error estimating properties from SMILES: {e}")
            return {}

    def find_similar_chemicals(
        self, smiles: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find chemically similar chronobiotics based on SMILES.

        Args:
            smiles: Reference SMILES string
            max_results: Maximum number of results to return

        Returns:
            List of similar chronobiotics with similarity scores
        """
        try:
            # This is a placeholder for actual chemical similarity search
            # In production, integrate with RDKit for fingerprint-based similarity

            all_chronobiotics = Chronobiotic.objects.exclude(smiles="").exclude(
                smiles__isnull=True
            )
            similar_chemicals = []

            # Simplified similarity based on SMILES length and composition
            ref_length = len(smiles)
            ref_atoms = self._count_atoms(smiles)

            for chronobiotic in all_chronobiotics:
                if not chronobiotic.smiles:
                    continue

                # Calculate simple similarity metrics
                target_length = len(chronobiotic.smiles)
                target_atoms = self._count_atoms(chronobiotic.smiles)

                # Length similarity
                length_similarity = 1 - abs(ref_length - target_length) / max(
                    ref_length, target_length
                )

                # Atom composition similarity
                common_atoms = 0
                total_atoms = 0
                all_atoms = set(list(ref_atoms.keys()) + list(target_atoms.keys()))

                for atom in all_atoms:
                    ref_count = ref_atoms.get(atom, 0)
                    target_count = target_atoms.get(atom, 0)
                    common_atoms += min(ref_count, target_count)
                    total_atoms += max(ref_count, target_count)

                if total_atoms > 0:
                    composition_similarity = common_atoms / total_atoms
                else:
                    composition_similarity = 0

                # Combined similarity score
                similarity_score = (
                    length_similarity * 0.3 + composition_similarity * 0.7
                ) * 100

                if similarity_score > 20:  # Threshold
                    similar_chemicals.append(
                        {
                            "chronobiotic": chronobiotic,
                            "similarity_score": round(similarity_score, 2),
                            "smiles": chronobiotic.smiles,
                            "molecular_formula": chronobiotic.molecula,
                            "length_similarity": round(length_similarity * 100, 2),
                            "composition_similarity": round(
                                composition_similarity * 100, 2
                            ),
                        }
                    )

            # Sort by similarity score
            similar_chemicals.sort(key=lambda x: x["similarity_score"], reverse=True)

            return similar_chemicals[:max_results]

        except Exception as e:
            logger.error(f"Error finding similar chemicals: {e}")
            return []

    def _count_atoms(self, smiles: str) -> Dict[str, int]:
        """Count atoms in SMILES string (simplified)."""
        atom_counts = {}

        # Simple regex to capture atoms (not perfect but works for basic cases)
        # This ignores charges, isotopes, and special cases
        atom_pattern = r"[A-Z][a-z]?"
        atoms = re.findall(atom_pattern, smiles)

        # Filter out non-atoms and brackets
        valid_atoms = [
            "C",
            "H",
            "O",
            "N",
            "S",
            "P",
            "F",
            "Cl",
            "Br",
            "I",
            "Na",
            "K",
            "Mg",
            "Ca",
            "Fe",
            "Zn",
            "Cu",
            "Ag",
            "Au",
        ]

        for atom in atoms:
            if atom in valid_atoms:
                atom_counts[atom] = atom_counts.get(atom, 0) + 1

        return atom_counts

    def validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Validate SMILES string format.

        Args:
            smiles: SMILES string to validate

        Returns:
            Validation results
        """
        validation = {
            "is_valid_format": False,
            "errors": [],
            "warnings": [],
            "parsed_elements": [],
        }

        try:
            # Basic validation rules
            if not smiles or not isinstance(smiles, str):
                validation["errors"].append("SMILES string is empty or not a string")
                return validation

            # Check for basic SMILES characters
            valid_chars = set("CHONSPFClBrI0123456789()[]=#+-.@/\\")
            invalid_chars = [c for c in smiles if c not in valid_chars]

            if invalid_chars:
                validation["errors"].append(
                    f'Invalid characters: {", ".join(set(invalid_chars))}'
                )

            # Check for balanced parentheses
            open_paren = smiles.count("(")
            close_paren = smiles.count(")")
            if open_paren != close_paren:
                validation["errors"].append(
                    f"Unbalanced parentheses: ({open_paren} open, {close_paren} close)"
                )

            # Check for balanced brackets
            open_bracket = smiles.count("[")
            close_bracket = smiles.count("]")
            if open_bracket != close_bracket:
                validation["errors"].append(
                    f"Unbalanced brackets: [{open_bracket} open, {close_bracket} close)"
                )

            # Basic atom detection
            atoms = self._count_atoms(smiles)
            if atoms:
                validation["parsed_elements"] = list(atoms.keys())
            else:
                validation["warnings"].append("No valid atoms detected")

            # Check for common patterns
            if "CC" in smiles and len(smiles) > 10:
                validation["warnings"].append(
                    "Contains carbon chain - may be organic compound"
                )

            # Check length
            if len(smiles) < 3:
                validation["errors"].append("SMILES too short")
            elif len(smiles) > 500:
                validation["warnings"].append(
                    "SMILES very long - may be complex molecule"
                )

            # Final validation
            validation["is_valid_format"] = len(validation["errors"]) == 0

            return validation

        except Exception as e:
            validation["errors"].append(f"Validation error: {str(e)}")
            return validation

    def get_chemical_classification(self, chronobiotic_id: int) -> Dict[str, Any]:
        """
        Get chemical classification based on structure and properties.

        Args:
            chronobiotic_id: ID of the chronobiotic

        Returns:
            Chemical classification data
        """
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
            properties = self.extract_chemical_properties(chronobiotic_id)

            classification = {
                "structural_classes": [],
                "functional_classes": [],
                "biological_classes": [],
                "estimated_class": None,
            }

            # Analyze SMILES for structural classes
            smiles = chronobiotic.smiles.upper()

            if "C1CCCCC1" in smiles or "C1=CC=CC=C1" in smiles:
                classification["structural_classes"].append("benzene_ring")
            if "N" in smiles and "C=O" in smiles:
                classification["structural_classes"].append("amide")
            if "COOH" in smiles or "C(=O)O" in smiles:
                classification["structural_classes"].append("carboxylic_acid")
            if "OH" in smiles and "C" in smiles:
                classification["structural_classes"].append("alcohol")
            if "NH2" in smiles:
                classification["structural_classes"].append("amine")

            # Get biological classes from database
            bioclasses = chronobiotic.classf.all()
            classification["biological_classes"] = [bc.nameclass for bc in bioclasses]

            # Determine estimated class based on properties
            if classification["structural_classes"]:
                classification["estimated_class"] = classification[
                    "structural_classes"
                ][0]
            elif classification["biological_classes"]:
                classification["estimated_class"] = classification[
                    "biological_classes"
                ][0]
            else:
                # Estimate based on molecular formula
                formula = chronobiotic.molecula
                if "C" in formula and "H" in formula:
                    if "O" in formula:
                        classification["estimated_class"] = "organic_compound"
                    elif "N" in formula:
                        classification["estimated_class"] = "nitrogenous_compound"
                    else:
                        classification["estimated_class"] = "hydrocarbon"
                elif "N" in formula and "H" in formula:
                    classification["estimated_class"] = "amine_derivative"

            return classification

        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return {}
        except Exception as e:
            logger.error(f"Error getting chemical classification: {e}")
            return {}

    def batch_analyze_chemicals(
        self, chronobiotic_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple chemicals in batch.

        Args:
            chronobiotic_ids: List of chronobiotic IDs

        Returns:
            List of analysis results
        """
        results = []

        for chronobiotic_id in chronobiotic_ids:
            try:
                analysis = {
                    "id": chronobiotic_id,
                    "properties": self.extract_chemical_properties(chronobiotic_id),
                    "classification": self.get_chemical_classification(chronobiotic_id),
                    "validation": (
                        self.validate_smiles(
                            Chronobiotic.objects.get(id=chronobiotic_id).smiles
                        )
                        if Chronobiotic.objects.filter(id=chronobiotic_id).exists()
                        else {}
                    ),
                }
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing chronobiotic {chronobiotic_id}: {e}")
                results.append({"id": chronobiotic_id, "error": str(e)})

        return results


# Global chemical model manager instance
chemical_model_manager = ChemicalModelManager()


def get_chemical_model_manager() -> ChemicalModelManager:
    """Get the global chemical model manager instance."""
    return chemical_model_manager
