"""
Query Executor for Complex Database Operations.

Provides methods for complex queries and data analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from django.db.models import Q, Count, Avg, Max, Min
from django.db import connection

from main.models import Chronobiotic, Targets, Mechanism, Bioclass, Effect
from .db_manager import get_db_manager

logger = logging.getLogger(__name__)


class QueryExecutor:
    """Executes complex database queries for agent system."""

    def __init__(self):
        self.db = get_db_manager()

    def find_chronobiotics_by_multiple_criteria(
        self, criteria: Dict[str, Any]
    ) -> List[Chronobiotic]:
        """
        Find chronobiotics by multiple criteria.

        Args:
            criteria: Dictionary of criteria, e.g., {
                'class': 'chronobiotics',
                'mechanism': 'relax',
                'target': 'T12345'
            }

        Returns:
            List of matching chronobiotics
        """
        try:
            query = Chronobiotic.objects.all()

            # Apply filters based on criteria
            if "name" in criteria:
                query = query.filter(gname__icontains=criteria["name"])

            if "class" in criteria:
                query = query.filter(classf__nameclass__iexact=criteria["class"])

            if "mechanism" in criteria:
                query = query.filter(
                    mechanisms__mechanismname__iexact=criteria["mechanism"]
                )

            if "target" in criteria:
                query = query.filter(target__targetsname__iexact=criteria["target"])

            if "effect" in criteria:
                query = query.filter(effect__Effectname__iexact=criteria["effect"])

            if "fdastatus" in criteria:
                query = query.filter(fdastatus__iexact=criteria["fdastatus"])

            # Remove duplicates
            query = query.distinct()

            return list(query)

        except Exception as e:
            logger.error(f"Error finding chronobiotics by criteria: {e}")
            return []

    def find_similar_chronobiotics(
        self, chronobiotic_id: int, similarity_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find chronobiotics similar to a given one.

        Args:
            chronobiotic_id: ID of reference chronobiotic
            similarity_fields: Fields to consider for similarity

        Returns:
            List of similar chronobiotics with similarity scores
        """
        try:
            reference = Chronobiotic.objects.get(id=chronobiotic_id)
            all_chronobiotics = Chronobiotic.objects.exclude(id=chronobiotic_id)

            if similarity_fields is None:
                similarity_fields = ["classf", "mechanisms", "target"]

            similar_items = []

            for chronobiotic in all_chronobiotics:
                score = 0
                matches = []

                # Compare classes
                if "classf" in similarity_fields:
                    ref_classes = set(reference.classf.all())
                    curr_classes = set(chronobiotic.classf.all())
                    common_classes = ref_classes.intersection(curr_classes)
                    if common_classes:
                        score += len(common_classes) * 10
                        matches.append(
                            f"Classes: {', '.join([c.nameclass for c in common_classes])}"
                        )

                # Compare mechanisms
                if "mechanisms" in similarity_fields:
                    ref_mechs = set(reference.mechanisms.all())
                    curr_mechs = set(chronobiotic.mechanisms.all())
                    common_mechs = ref_mechs.intersection(curr_mechs)
                    if common_mechs:
                        score += len(common_mechs) * 5
                        matches.append(
                            f"Mechanisms: {', '.join([m.mechanismname for m in common_mechs])}"
                        )

                # Compare targets
                if "target" in similarity_fields:
                    ref_targets = set(reference.target.all())
                    curr_targets = set(chronobiotic.target.all())
                    common_targets = ref_targets.intersection(curr_targets)
                    if common_targets:
                        score += len(common_targets) * 8
                        matches.append(
                            f"Targets: {', '.join([t.targetsname for t in common_targets])}"
                        )

                # Compare effects
                if "effect" in similarity_fields:
                    ref_effects = set(reference.effect.all())
                    curr_effects = set(chronobiotic.effect.all())
                    common_effects = ref_effects.intersection(curr_effects)
                    if common_effects:
                        score += len(common_effects) * 3
                        matches.append(
                            f"Effects: {', '.join([e.Effectname for e in common_effects])}"
                        )

                if score > 0:
                    similar_items.append(
                        {
                            "chronobiotic": chronobiotic,
                            "score": score,
                            "matches": matches,
                            "similarity_percentage": min(score, 100),
                        }
                    )

            # Sort by similarity score
            similar_items.sort(key=lambda x: x["score"], reverse=True)

            return similar_items[:10]  # Return top 10

        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return []
        except Exception as e:
            logger.error(f"Error finding similar chronobiotics: {e}")
            return []

    def get_chronobiotics_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about chronobiotics."""
        stats = {}

        try:
            # Basic counts
            total = Chronobiotic.objects.count()
            stats["total"] = total

            # Count by FDA status
            fda_stats = {}
            for status in Chronobiotic.objects.values_list(
                "fdastatus", flat=True
            ).distinct():
                if status:
                    count = Chronobiotic.objects.filter(fdastatus=status).count()
                    fda_stats[status] = count
            stats["by_fda_status"] = fda_stats

            # Top mechanisms
            mechanism_stats = []
            for mechanism in Mechanism.objects.annotate(
                chrono_count=Count("chronobiotic")
            ).order_by("-chrono_count")[:10]:
                mechanism_stats.append(
                    {
                        "mechanism": mechanism.mechanismname,
                        "count": mechanism.chrono_count,
                    }
                )
            stats["top_mechanisms"] = mechanism_stats

            # Top targets
            target_stats = []
            for target in Targets.objects.annotate(
                chrono_count=Count("chronobiotic")
            ).order_by("-chrono_count")[:10]:
                target_stats.append(
                    {
                        "target": target.targetsname,
                        "fullname": target.targetsfullname,
                        "count": target.chrono_count,
                    }
                )
            stats["top_targets"] = target_stats

            # Top bioclasses
            class_stats = []
            for bioclass in Bioclass.objects.annotate(
                chrono_count=Count("chronobiotic")
            ).order_by("-chrono_count")[:10]:
                class_stats.append(
                    {"class": bioclass.nameclass, "count": bioclass.chrono_count}
                )
            stats["top_classes"] = class_stats

            # Distribution by mechanism count
            mechanism_count_dist = {}
            for chronobiotic in Chronobiotic.objects.all():
                count = chronobiotic.mechanisms.count()
                mechanism_count_dist[count] = mechanism_count_dist.get(count, 0) + 1
            stats["mechanism_count_distribution"] = mechanism_count_dist

            # Distribution by target count
            target_count_dist = {}
            for chronobiotic in Chronobiotic.objects.all():
                count = chronobiotic.target.count()
                target_count_dist[count] = target_count_dist.get(count, 0) + 1
            stats["target_count_distribution"] = target_count_dist

            return stats

        except Exception as e:
            logger.error(f"Error getting chronobiotics statistics: {e}")
            return {}

    def find_chronobiotics_by_chemical_pattern(
        self, pattern: str
    ) -> List[Chronobiotic]:
        """
        Find chronobiotics by chemical pattern (SMILES substructure).
        This is a placeholder - actual chemical similarity search would use RDKit.

        Args:
            pattern: Chemical pattern to search for

        Returns:
            List of matching chronobiotics
        """
        try:
            # For now, do simple text search in SMILES
            # In production, integrate with RDKit for actual substructure search
            chronobiotics = Chronobiotic.objects.filter(
                Q(smiles__icontains=pattern)
                | Q(molecula__icontains=pattern)
                | Q(iupacname__icontains=pattern)
            ).distinct()

            return list(chronobiotics)

        except Exception as e:
            logger.error(f"Error finding chronobiotics by chemical pattern: {e}")
            return []

    def get_relationships_network(self, chronobiotic_id: int) -> Dict[str, Any]:
        """
        Get relationship network for a chronobiotic.

        Returns network of related entities (targets, mechanisms, classes, etc.)
        """
        try:
            chronobiotic = Chronobiotic.objects.get(id=chronobiotic_id)

            network = {
                "nodes": [],
                "edges": [],
                "center": {
                    "id": chronobiotic.id,
                    "label": chronobiotic.gname,
                    "type": "chronobiotic",
                },
            }

            # Add chronobiotic as central node
            network["nodes"].append(
                {
                    "id": f"chronobiotic_{chronobiotic.id}",
                    "label": chronobiotic.gname,
                    "type": "chronobiotic",
                    "size": 20,
                }
            )

            node_id = 1

            # Add targets
            for target in chronobiotic.target.all():
                node_key = f"target_{target.id}"
                network["nodes"].append(
                    {
                        "id": node_key,
                        "label": target.targetsname,
                        "type": "target",
                        "size": 10,
                    }
                )
                network["edges"].append(
                    {
                        "source": f"chronobiotic_{chronobiotic.id}",
                        "target": node_key,
                        "type": "targets",
                    }
                )
                node_id += 1

            # Add mechanisms
            for mechanism in chronobiotic.mechanisms.all():
                node_key = f"mechanism_{mechanism.id}"
                network["nodes"].append(
                    {
                        "id": node_key,
                        "label": mechanism.mechanismname,
                        "type": "mechanism",
                        "size": 8,
                    }
                )
                network["edges"].append(
                    {
                        "source": f"chronobiotic_{chronobiotic.id}",
                        "target": node_key,
                        "type": "mechanism",
                    }
                )
                node_id += 1

            # Add classes
            for bioclass in chronobiotic.classf.all():
                node_key = f"class_{bioclass.id}"
                network["nodes"].append(
                    {
                        "id": node_key,
                        "label": bioclass.nameclass,
                        "type": "class",
                        "size": 12,
                    }
                )
                network["edges"].append(
                    {
                        "source": f"chronobiotic_{chronobiotic.id}",
                        "target": node_key,
                        "type": "class",
                    }
                )
                node_id += 1

            # Add effects
            for effect in chronobiotic.effect.all():
                node_key = f"effect_{effect.id}"
                network["nodes"].append(
                    {
                        "id": node_key,
                        "label": effect.Effectname,
                        "type": "effect",
                        "size": 6,
                    }
                )
                network["edges"].append(
                    {
                        "source": f"chronobiotic_{chronobiotic.id}",
                        "target": node_key,
                        "type": "effect",
                    }
                )
                node_id += 1

            return network

        except Chronobiotic.DoesNotExist:
            logger.warning(f"Chronobiotic ID {chronobiotic_id} not found")
            return {"nodes": [], "edges": []}
        except Exception as e:
            logger.error(f"Error getting relationships network: {e}")
            return {"nodes": [], "edges": []}

    def execute_advanced_query(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute advanced pre-defined queries.

        Args:
            query_type: Type of query to execute
            **kwargs: Query parameters

        Returns:
            Query results
        """
        query_handlers = {
            "chronobiotics_with_multiple_targets": self._query_multiple_targets,
            "chronobiotics_by_class_and_mechanism": self._query_class_and_mechanism,
            "target_frequency": self._query_target_frequency,
            "mechanism_combinations": self._query_mechanism_combinations,
        }

        if query_type not in query_handlers:
            raise ValueError(f"Unknown query type: {query_type}")

        return query_handlers[query_type](**kwargs)

    def _query_multiple_targets(self, min_targets: int = 2) -> List[Dict[str, Any]]:
        """Find chronobiotics targeting multiple targets."""
        try:
            # Raw SQL for efficiency with ManyToMany counts
            query = """
            SELECT 
                c.id,
                c.gname,
                c.molecula,
                COUNT(ct.target_id) as target_count,
                GROUP_CONCAT(t.targetsname) as target_names
            FROM chronobiotic c
            JOIN chronobiotic_target ct ON c.id = ct.chronobiotic_id
            JOIN target t ON ct.target_id = t.id
            GROUP BY c.id
            HAVING target_count >= %s
            ORDER BY target_count DESC
            """

            results = []
            with connection.cursor() as cursor:
                cursor.execute(query, [min_targets])
                for row in cursor.fetchall():
                    results.append(
                        {
                            "id": row[0],
                            "gname": row[1],
                            "molecula": row[2],
                            "target_count": row[3],
                            "target_names": row[4].split(",") if row[4] else [],
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error in multiple targets query: {e}")
            return []

    def _query_class_and_mechanism(
        self, class_name: str, mechanism_name: str
    ) -> List[Dict[str, Any]]:
        """Find chronobiotics by class and mechanism."""
        try:
            chronobiotics = Chronobiotic.objects.filter(
                classf__nameclass__iexact=class_name,
                mechanisms__mechanismname__iexact=mechanism_name,
            ).distinct()

            results = []
            for chronobiotic in chronobiotics:
                results.append(
                    {
                        "id": chronobiotic.id,
                        "gname": chronobiotic.gname,
                        "smiles": chronobiotic.smiles,
                        "description": chronobiotic.description,
                        "fdastatus": chronobiotic.fdastatus,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error in class and mechanism query: {e}")
            return []

    def _query_target_frequency(self) -> List[Dict[str, Any]]:
        """Get frequency of targets across all chronobiotics."""
        try:
            query = """
            SELECT 
                t.targetsname,
                t.targetsfullname,
                COUNT(ct.chronobiotic_id) as frequency,
                GROUP_CONCAT(DISTINCT c.gname) as chronobiotic_names
            FROM target t
            LEFT JOIN chronobiotic_target ct ON t.id = ct.target_id
            LEFT JOIN chronobiotic c ON ct.chronobiotic_id = c.id
            GROUP BY t.id
            HAVING frequency > 0
            ORDER BY frequency DESC
            """

            results = []
            with connection.cursor() as cursor:
                cursor.execute(query)
                for row in cursor.fetchall():
                    results.append(
                        {
                            "target": row[0],
                            "fullname": row[1],
                            "frequency": row[2],
                            "chronobiotic_names": row[3].split(",") if row[3] else [],
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error in target frequency query: {e}")
            return []

    def _query_mechanism_combinations(self) -> List[Dict[str, Any]]:
        """Find common mechanism combinations."""
        try:
            # This is a complex query - simplified version
            query = """
            SELECT 
                m1.mechanismname as mechanism1,
                m2.mechanismname as mechanism2,
                COUNT(DISTINCT c.id) as co_occurrence
            FROM mechanism m1
            JOIN chronobiotic_mechanisms cm1 ON m1.id = cm1.mechanism_id
            JOIN chronobiotic c ON cm1.chronobiotic_id = c.id
            JOIN chronobiotic_mechanisms cm2 ON c.id = cm2.chronobiotic_id
            JOIN mechanism m2 ON cm2.mechanism_id = m2.id
            WHERE m1.id < m2.id
            GROUP BY m1.id, m2.id
            HAVING co_occurrence > 0
            ORDER BY co_occurrence DESC
            LIMIT 20
            """

            results = []
            with connection.cursor() as cursor:
                cursor.execute(query)
                for row in cursor.fetchall():
                    results.append(
                        {
                            "mechanism1": row[0],
                            "mechanism2": row[1],
                            "co_occurrence": row[2],
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error in mechanism combinations query: {e}")
            return []


# Singleton instance
query_executor = QueryExecutor()


def get_query_executor() -> QueryExecutor:
    """Get the global query executor instance."""
    return query_executor
