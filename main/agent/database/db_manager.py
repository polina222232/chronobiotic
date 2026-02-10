"""
Database Manager for Chronobiotic Agent System.

Provides unified interface for database operations across all agent modules.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Type
from django.db import connection, transaction
from django.db.models import Model, QuerySet
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned

from main.models import (
    Chronobiotic,
    Synonyms,
    Articles,
    Targets,
    Effect,
    Mechanism,
    Bioclass,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Main database manager for agent operations."""

    def __init__(self):
        self._connected = False
        self._models = {
            "chronobiotic": Chronobiotic,
            "synonyms": Synonyms,
            "articles": Articles,
            "targets": Targets,
            "effect": Effect,
            "mechanism": Mechanism,
            "bioclass": Bioclass,
        }
        self.connect()

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            # Test connection
            connection.ensure_connection()
            self._connected = True
            logger.info("Database connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        """Check if database connection is active."""
        return self._connected

    def get_model(self, model_name: str) -> Type[Model]:
        """Get Django model by name."""
        if model_name not in self._models:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(self._models.keys())}"
            )
        return self._models[model_name]

    # ==================== CHRONOBIOTIC OPERATIONS ====================

    def get_chronobiotic_by_name(self, name: str) -> Optional[Chronobiotic]:
        """Get chronobiotic by its generic name."""
        try:
            return Chronobiotic.objects.get(gname__iexact=name)
        except Chronobiotic.DoesNotExist:
            logger.debug(f"Chronobiotic '{name}' not found")
            return None
        except Exception as e:
            logger.error(f"Error fetching chronobiotic '{name}': {e}")
            return None

    def get_chronobiotic_by_smiles(self, smiles: str) -> Optional[Chronobiotic]:
        """Get chronobiotic by SMILES string."""
        try:
            return Chronobiotic.objects.get(smiles=smiles)
        except Chronobiotic.DoesNotExist:
            logger.debug(f"Chronobiotic with SMILES '{smiles}' not found")
            return None
        except Exception as e:
            logger.error(f"Error fetching chronobiotic by SMILES: {e}")
            return None

    def search_chronobiotics(self, query: str, field: str = "gname") -> QuerySet:
        """
        Search chronobiotics by various fields.

        Args:
            query: Search query
            field: Field to search in ('gname', 'molecula', 'iupacname', 'description')

        Returns:
            QuerySet of matching chronobiotics
        """
        valid_fields = ["gname", "molecula", "iupacname", "description"]
        if field not in valid_fields:
            raise ValueError(f"Invalid field '{field}'. Must be one of: {valid_fields}")

        filter_kwargs = {f"{field}__icontains": query}
        return Chronobiotic.objects.filter(**filter_kwargs)

    def get_chronobiotics_by_class(self, class_name: str) -> QuerySet:
        """Get all chronobiotics belonging to a specific bioclass."""
        try:
            bioclass = Bioclass.objects.get(nameclass__iexact=class_name)
            return Chronobiotic.objects.filter(classf=bioclass)
        except Bioclass.DoesNotExist:
            logger.warning(f"Bioclass '{class_name}' not found")
            return Chronobiotic.objects.none()
        except Exception as e:
            logger.error(f"Error fetching chronobiotics by class: {e}")
            return Chronobiotic.objects.none()

    def get_chronobiotics_by_target(self, target_name: str) -> QuerySet:
        """Get all chronobiotics targeting a specific target."""
        try:
            target = Targets.objects.get(targetsname__iexact=target_name)
            return Chronobiotic.objects.filter(target=target)
        except Targets.DoesNotExist:
            logger.warning(f"Target '{target_name}' not found")
            return Chronobiotic.objects.none()
        except Exception as e:
            logger.error(f"Error fetching chronobiotics by target: {e}")
            return Chronobiotic.objects.none()

    def get_chronobiotics_by_mechanism(self, mechanism_name: str) -> QuerySet:
        """Get all chronobiotics with a specific mechanism."""
        try:
            mechanism = Mechanism.objects.get(mechanismname__iexact=mechanism_name)
            return Chronobiotic.objects.filter(mechanisms=mechanism)
        except Mechanism.DoesNotExist:
            logger.warning(f"Mechanism '{mechanism_name}' not found")
            return Chronobiotic.objects.none()
        except Exception as e:
            logger.error(f"Error fetching chronobiotics by mechanism: {e}")
            return Chronobiotic.objects.none()

    # ==================== RELATED DATA OPERATIONS ====================

    def get_synonyms_for_chronobiotic(self, chronobiotic_id: int) -> List[str]:
        """Get all synonyms for a chronobiotic."""
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
            synonyms = Synonyms.objects.filter(originalbiotic=chronobiotic)
            return [syn.synonymsmname for syn in synonyms]
        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return []
        except Exception as e:
            logger.error(f"Error fetching synonyms: {e}")
            return []

    def get_articles_for_chronobiotic(self, chronobiotic_id: int) -> QuerySet:
        """Get all articles related to a chronobiotic."""
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
            return chronobiotic.articles.all()
        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return Articles.objects.none()
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return Articles.objects.none()

    def get_targets_for_chronobiotic(self, chronobiotic_id: int) -> QuerySet:
        """Get all targets for a chronobiotic."""
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
            return chronobiotic.target.all()
        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return Targets.objects.none()
        except Exception as e:
            logger.error(f"Error fetching targets: {e}")
            return Targets.objects.none()

    def get_mechanisms_for_chronobiotic(self, chronobiotic_id: int) -> QuerySet:
        """Get all mechanisms for a chronobiotic."""
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
            return chronobiotic.mechanisms.all()
        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return Mechanism.objects.none()
        except Exception as e:
            logger.error(f"Error fetching mechanisms: {e}")
            return Mechanism.objects.none()

    def get_effects_for_chronobiotic(self, chronobiotic_id: int) -> QuerySet:
        """Get all effects for a chronobiotic."""
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
            return chronobiotic.effect.all()
        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return Effect.objects.none()
        except Exception as e:
            logger.error(f"Error fetching effects: {e}")
            return Effect.objects.none()

    def get_classes_for_chronobiotic(self, chronobiotic_id: int) -> QuerySet:
        """Get all bioclasses for a chronobiotic."""
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)
            return chronobiotic.classf.all()
        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return Bioclass.objects.none()
        except Exception as e:
            logger.error(f"Error fetching classes: {e}")
            return Bioclass.objects.none()

    # ==================== STATISTICS & METADATA ====================

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            "total_chronobiotics": Chronobiotic.objects.count(),
            "total_articles": Articles.objects.count(),
            "total_targets": Targets.objects.count(),
            "total_mechanisms": Mechanism.objects.count(),
            "total_effects": Effect.objects.count(),
            "total_bioclasses": Bioclass.objects.count(),
            "total_synonyms": Synonyms.objects.count(),
        }

        # Add top classes
        classes_stats = {}
        for bioclass in Bioclass.objects.all():
            count = Chronobiotic.objects.filter(classf=bioclass).count()
            classes_stats[bioclass.nameclass] = count
        stats["chronobiotics_by_class"] = classes_stats

        # Add top mechanisms
        mechanisms_stats = {}
        for mechanism in Mechanism.objects.all():
            count = Chronobiotic.objects.filter(mechanisms=mechanism).count()
            mechanisms_stats[mechanism.mechanismname] = count
        stats["chronobiotics_by_mechanism"] = mechanisms_stats

        return stats

    def get_chronobiotic_full_info(self, chronobiotic_id: int) -> Dict[str, Any]:
        """Get complete information about a chronobiotic."""
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)

            # Get all related data
            synonyms = self.get_synonyms_for_chronobiotic(chronobiotic_id)
            articles = list(
                self.get_articles_for_chronobiotic(chronobiotic_id).values()
            )
            targets = list(self.get_targets_for_chronobiotic(chronobiotic_id).values())
            mechanisms = list(
                self.get_mechanisms_for_chronobiotic(chronobiotic_id).values()
            )
            effects = list(self.get_effects_for_chronobiotic(chronobiotic_id).values())
            classes = list(self.get_classes_for_chronobiotic(chronobiotic_id).values())

            # Build comprehensive info dict
            info = {
                "id": chronobiotic.id,
                "gname": chronobiotic.gname,
                "smiles": chronobiotic.smiles,
                "linkname": chronobiotic.linkname,
                "molecula": chronobiotic.molecula,
                "iupacname": chronobiotic.iupacname,
                "description": chronobiotic.description,
                "fdastatus": chronobiotic.fdastatus,
                "linkslists": chronobiotic.linkslists,
                "pubchem": chronobiotic.pubchem,
                "chemspider": chronobiotic.chemspider,
                "drugbank": chronobiotic.drugbank,
                "chebi": chronobiotic.chebi,
                "uniprot": chronobiotic.uniprot,
                "kegg": chronobiotic.kegg,
                "selleckchem": chronobiotic.selleckchem,
                "synonyms": synonyms,
                "articles": articles,
                "targets": targets,
                "mechanisms": mechanisms,
                "effects": effects,
                "classes": classes,
            }

            return info

        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return {}
        except Exception as e:
            logger.error(f"Error fetching chronobiotic full info: {e}")
            return {}

    # ==================== TRANSACTION MANAGEMENT ====================

    @transaction.atomic
    def create_chronobiotic(self, data: Dict[str, Any]) -> Optional[Chronobiotic]:
        """Create a new chronobiotic record."""
        try:
            # Extract ManyToMany fields
            articles_data = data.pop("articles", [])
            classf_data = data.pop("classf", [])
            mechanisms_data = data.pop("mechanisms", [])
            target_data = data.pop("target", [])
            effect_data = data.pop("effect", [])

            # Create the chronobiotic
            chronobiotic = Chronobiotic.objects.create(**data)

            # Add ManyToMany relationships
            if articles_data:
                articles = Articles.objects.filter(id__in=articles_data)
                chronobiotic.articles.add(*articles)

            if classf_data:
                classes = Bioclass.objects.filter(id__in=classf_data)
                chronobiotic.classf.add(*classes)

            if mechanisms_data:
                mechanisms = Mechanism.objects.filter(id__in=mechanisms_data)
                chronobiotic.mechanisms.add(*mechanisms)

            if target_data:
                targets = Targets.objects.filter(id__in=target_data)
                chronobiotic.target.add(*targets)

            if effect_data:
                effects = Effect.objects.filter(id__in=effect_data)
                chronobiotic.effect.add(*effects)

            logger.info(f"Created chronobiotic: {chronobiotic.gname}")
            return chronobiotic

        except Exception as e:
            logger.error(f"Error creating chronobiotic: {e}")
            return None

    @transaction.atomic
    def update_chronobiotic(
        self, chronobiotic_id: int, data: Dict[str, Any]
    ) -> Optional[Chronobiotic]:
        """Update an existing chronobiotic."""
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)

            # Update simple fields
            for field, value in data.items():
                if hasattr(chronobiotic, field) and field not in [
                    "articles",
                    "classf",
                    "mechanisms",
                    "target",
                    "effect",
                ]:
                    setattr(chronobiotic, field, value)

            chronobiotic.save()
            logger.info(f"Updated chronobiotic ID {chronobiotic_id}")
            return chronobiotic

        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found for update")
            return None
        except Exception as e:
            logger.error(f"Error updating chronobiotic: {e}")
            return None

    # ==================== QUERY EXECUTION ====================

    def execute_raw_query(
        self, query: str, params: List[Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute raw SQL query."""
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params or [])
                columns = [col[0] for col in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return results
        except Exception as e:
            logger.error(f"Error executing raw query: {e}")
            return []

    def close(self):
        """Close database connection."""
        try:
            connection.close()
            self._connected = False
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")


# Singleton instance for global use
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager
