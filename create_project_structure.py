import os
from pathlib import Path


class ProjectCreator:
    def __init__(self, base_dir="chronobiotic"):
        self.base_dir = Path(base_dir)
    
    def create_directory(self, path):
        """Создает директорию если она не существует"""
        full_path = self.base_dir / path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Создана директория: {path}")
            return True
        return False
    
    def create_file(self, path, content=""):
        """Создает файл если он не существует"""
        full_path = self.base_dir / path
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Создан файл: {path}")
            return True
        print(f"✓ Уже существует: {path}")
        return False
    
    def create_structure(self):
        print("=" * 60)
        print("Создание структуры проекта Chronobiotic...")
        print("=" * 60)
        
        self._create_directories()
        self._create_init_files()
        self._create_config_files()
        self._create_main_files()
        self._create_agent_files()
        self._create_api_files()
        self._create_migration_files()
        self._create_static_files()
        self._create_template_files()
        self._create_media_files()
        self._create_requirements_files()
        self._create_test_files()
        self._create_utils_files()
        self._create_root_files()
        
        print("\n" + "=" * 60)
        print("Структура проекта успешно создана!")
        print("=" * 60)
    
    def _create_directories(self):
        """Создает все директории проекта"""
        directories = [
            ".idea",
            "config",
            "fixtures",
            "main/agent/agents/analysis_agents",
            "main/agent/agents/assistant_agents",
            "main/agent/agents/chronobiotics_agents",
            "main/agent/agents/citation_agents",
            "main/agent/agents/data_agents",
            "main/agent/agents/research_agents",
            "main/agent/analysis",
            "main/agent/chat",
            "main/agent/chem/analysis",
            "main/agent/chem/db",
            "main/agent/chem/external",
            "main/agent/chem/img",
            "main/agent/chem/parser",
            "main/agent/chem/util",
            "main/agent/chem/validation",
            "main/agent/citation",
            "main/agent/core",
            "main/agent/database",
            "main/agent/kag/algorithms",
            "main/agent/kag/models",
            "main/agent/kag/queries",
            "main/agent/kag/storage",
            "main/agent/kag/utils",
            "main/agent/llm/fine_tuning",
            "main/agent/llm/models",
            "main/agent/llm/multimodal_agents/embeddings",
            "main/agent/llm/multimodal_agents/fusion",
            "main/agent/llm/multimodal_agents/vision",
            "main/agent/llm/multimodal_llm",
            "main/agent/llm/optimization",
            "main/agent/llm/prompts",
            "main/agent/llm/tools",
            "main/agent/management/commands/agent_commands",
            "main/agent/management/commands/data_commands",
            "main/agent/management/commands/kag_commands",
            "main/agent/management/commands/llm_commands",
            "main/agent/management/commands/rag_commands",
            "main/agent/management/commands/system_commands",
            "main/agent/memory",
            "main/agent/parallel",
            "main/agent/rag/chunking",
            "main/agent/rag/embeddings",
            "main/agent/rag/knowledge_base",
            "main/agent/rag/reranking",
            "main/agent/rag/retrievers",
            "main/agent/rag/vector_store",
            "main/agent/response",
            "main/agent/search",
            "main/agent/tasks/agent_tasks",
            "main/agent/tasks/background_tasks",
            "main/agent/tasks/chem_tasks",
            "main/agent/tasks/kag_tasks",
            "main/agent/tasks/llm_tasks",
            "main/agent/tasks/periodic_tasks",
            "main/agent/tasks/rag_tasks",
            "main/agent/utils",
            "main/agent/web",
            "main/api/v1/agents",
            "main/api/v1/chat",
            "main/api/v1/chemical",
            "main/api/v1/data",
            "main/api/v1/kag",
            "main/api/v1/rag",
            "main/api/websocket",
            "main/migrations",
            "main/static/css",
            "main/static/js",
            "main/static/images",
            "main/templates/main",
            "media/chemical_structures",
            "media/documents",
            "media/user_uploads",
            "requirements",
            "tests/benchmarks",
            "tests/fixtures/test_images",
            "tests/test_agent",
            "tests/test_chem",
            "tests/test_kag",
            "tests/test_llm",
            "tests/test_rag",
            "tests/test_tasks",
            "tests/test_utils",
            "utils/data_processing",
            "utils/error_handling",
            "utils/file_handling",
            "utils/logging",
            "utils/monitoring",
            "utils/network",
            "utils/security",
            "utils/time",
        ]
        
        for directory in directories:
            self.create_directory(directory)
    
    def _create_init_files(self):
        """Создает все __init__.py файлы"""
        init_files = [
            "__init__.py",
            "config/__init__.py",
            "main/__init__.py",
            "main/agent/__init__.py",
            "main/agent/agents/__init__.py",
            "main/agent/agents/analysis_agents/__init__.py",
            "main/agent/agents/assistant_agents/__init__.py",
            "main/agent/agents/chronobiotics_agents/__init__.py",
            "main/agent/agents/citation_agents/__init__.py",
            "main/agent/agents/data_agents/__init__.py",
            "main/agent/agents/research_agents/__init__.py",
            "main/agent/analysis/__init__.py",
            "main/agent/chat/__init__.py",
            "main/agent/chem/__init__.py",
            "main/agent/chem/analysis/__init__.py",
            "main/agent/chem/db/__init__.py",
            "main/agent/chem/external/__init__.py",
            "main/agent/chem/img/__init__.py",
            "main/agent/chem/parser/__init__.py",
            "main/agent/chem/util/__init__.py",
            "main/agent/chem/validation/__init__.py",
            "main/agent/citation/__init__.py",
            "main/agent/core/__init__.py",
            "main/agent/database/__init__.py",
            "main/agent/kag/__init__.py",
            "main/agent/kag/algorithms/__init__.py",
            "main/agent/kag/models/__init__.py",
            "main/agent/kag/queries/__init__.py",
            "main/agent/kag/storage/__init__.py",
            "main/agent/kag/utils/__init__.py",
            "main/agent/llm/__init__.py",
            "main/agent/llm/fine_tuning/__init__.py",
            "main/agent/llm/models/__init__.py",
            "main/agent/llm/multimodal_agents/__init__.py",
            "main/agent/llm/multimodal_agents/embeddings/__init__.py",
            "main/agent/llm/multimodal_agents/fusion/__init__.py",
            "main/agent/llm/multimodal_agents/vision/__init__.py",
            "main/agent/llm/multimodal_llm/__init__.py",
            "main/agent/llm/optimization/__init__.py",
            "main/agent/llm/prompts/__init__.py",
            "main/agent/llm/tools/__init__.py",
            "main/agent/management/__init__.py",
            "main/agent/management/commands/__init__.py",
            "main/agent/management/commands/agent_commands/__init__.py",
            "main/agent/management/commands/data_commands/__init__.py",
            "main/agent/management/commands/kag_commands/__init__.py",
            "main/agent/management/commands/llm_commands/__init__.py",
            "main/agent/management/commands/rag_commands/__init__.py",
            "main/agent/management/commands/system_commands/__init__.py",
            "main/agent/memory/__init__.py",
            "main/agent/parallel/__init__.py",
            "main/agent/rag/__init__.py",
            "main/agent/rag/chunking/__init__.py",
            "main/agent/rag/embeddings/__init__.py",
            "main/agent/rag/knowledge_base/__init__.py",
            "main/agent/rag/reranking/__init__.py",
            "main/agent/rag/retrievers/__init__.py",
            "main/agent/rag/vector_store/__init__.py",
            "main/agent/response/__init__.py",
            "main/agent/search/__init__.py",
            "main/agent/tasks/__init__.py",
            "main/agent/tasks/agent_tasks/__init__.py",
            "main/agent/tasks/background_tasks/__init__.py",
            "main/agent/tasks/chem_tasks/__init__.py",
            "main/agent/tasks/kag_tasks/__init__.py",
            "main/agent/tasks/llm_tasks/__init__.py",
            "main/agent/tasks/periodic_tasks/__init__.py",
            "main/agent/tasks/rag_tasks/__init__.py",
            "main/agent/utils/__init__.py",
            "main/agent/web/__init__.py",
            "main/api/__init__.py",
            "main/api/v1/__init__.py",
            "main/api/v1/agents/__init__.py",
            "main/api/v1/chat/__init__.py",
            "main/api/v1/chemical/__init__.py",
            "main/api/v1/data/__init__.py",
            "main/api/v1/kag/__init__.py",
            "main/api/v1/rag/__init__.py",
            "main/api/websocket/__init__.py",
            "tests/__init__.py",
            "tests/benchmarks/__init__.py",
            "tests/fixtures/__init__.py",
            "tests/test_agent/__init__.py",
            "tests/test_chem/__init__.py",
            "tests/test_kag/__init__.py",
            "tests/test_llm/__init__.py",
            "tests/test_rag/__init__.py",
            "tests/test_tasks/__init__.py",
            "tests/test_utils/__init__.py",
            "utils/__init__.py",
            "utils/data_processing/__init__.py",
            "utils/error_handling/__init__.py",
            "utils/file_handling/__init__.py",
            "utils/logging/__init__.py",
            "utils/monitoring/__init__.py",
            "utils/network/__init__.py",
            "utils/security/__init__.py",
            "utils/time/__init__.py",
        ]
        
        for file_path in init_files:
            self.create_file(file_path)
    
    def _create_config_files(self):
        """Создает конфигурационные файлы"""
        config_files = [
            "config/settings.py",
            "config/settings_dev.py",
            "config/settings_prod.py",
            "config/settings_test.py",
            "config/urls.py",
            "config/wsgi.py",
            "config/asgi.py",
        ]
        
        for file_path in config_files:
            self.create_file(file_path, "# Конфигурационный файл\n\n")
    
    def _create_main_files(self):
        """Создает основные файлы проекта"""
        main_files = [
            "main/admin.py",
            "main/apps.py",
            "main/models.py",
            "main/tests.py",
            "main/urls.py",
            "main/views.py",
        ]
        
        for file_path in main_files:
            self.create_file(file_path, "# Django файл\n\n")
    
    def _create_agent_files(self):
        """Создает файлы агентской системы"""
        agent_files = [
            # Основные файлы агента
            "main/agent/agent_core.py",
            "main/agent/agent_manager.py",
            "main/agent/agent_monitor.py",
            "main/agent/chat_interface.py",
            "main/agent/chronobiotics_agent.py",
            "main/agent/citation_system.py",
            "main/agent/parallel_executor.py",
            "main/agent/response_formatter.py",
            "main/agent/result_aggregator.py",
            "main/agent/task_dispatcher.py",
            
            # Агенты анализа
            "main/agent/agents/base_agent.py",
            "main/agent/agents/analysis_agents/chemical_analyzer.py",
            "main/agent/agents/analysis_agents/efficacy_evaluator.py",
            "main/agent/agents/analysis_agents/interaction_analyzer.py",
            "main/agent/agents/analysis_agents/property_predictor.py",
            "main/agent/agents/analysis_agents/similarity_finder.py",
            "main/agent/agents/analysis_agents/toxicity_estimator.py",
            
            # Агенты-ассистенты
            "main/agent/agents/assistant_agents/chat_agent.py",
            "main/agent/agents/assistant_agents/explanation_agent.py",
            "main/agent/agents/assistant_agents/qa_agent.py",
            "main/agent/agents/assistant_agents/recommendation_agent.py",
            "main/agent/agents/assistant_agents/summarizer_agent.py",
            
            # Агенты хронобиотиков
            "main/agent/agents/chronobiotics_agents/chronobiology_expert.py",
            "main/agent/agents/chronobiotics_agents/chronobiotics_searcher.py",
            "main/agent/agents/chronobiotics_agents/clinical_data_finder.py",
            "main/agent/agents/chronobiotics_agents/literature_miner.py",
            "main/agent/agents/chronobiotics_agents/mechanism_researcher.py",
            "main/agent/agents/chronobiotics_agents/substance_analyzer.py",
            
            # Агенты цитирования
            "main/agent/agents/citation_agents/bibliography_builder.py",
            "main/agent/agents/citation_agents/citation_extractor.py",
            "main/agent/agents/citation_agents/reference_formatter.py",
            "main/agent/agents/citation_agents/source_tracker.py",
            "main/agent/agents/citation_agents/source_validator.py",
            
            # Агенты данных
            "main/agent/agents/data_agents/content_analyzer.py",
            "main/agent/agents/data_agents/data_storage.py",
            "main/agent/agents/data_agents/data_validator.py",
            "main/agent/agents/data_agents/database_searcher.py",
            "main/agent/agents/data_agents/link_follower.py",
            "main/agent/agents/data_agents/web_scraper.py",
            
            # Исследовательские агенты
            "main/agent/agents/research_agents/clinical_trial_finder.py",
            "main/agent/agents/research_agents/hypothesis_generator.py",
            "main/agent/agents/research_agents/literature_reviewer.py",
            "main/agent/agents/research_agents/mechanism_investigator.py",
            "main/agent/agents/research_agents/patent_searcher.py",
            
            # Анализ
            "main/agent/analysis/analysis_engine.py",
            "main/agent/analysis/analysis_validator.py",
            "main/agent/analysis/chemical_analyzer.py",
            "main/agent/analysis/chronobiotics_analyzer.py",
            "main/agent/analysis/data_processor.py",
            "main/agent/analysis/insight_extractor.py",
            "main/agent/analysis/pattern_finder.py",
            "main/agent/analysis/text_analyzer.py",
            
            # Чат
            "main/agent/chat/chat_engine.py",
            "main/agent/chat/chat_formatter.py",
            "main/agent/chat/chat_history.py",
            "main/agent/chat/conversation_manager.py",
            "main/agent/chat/message_handler.py",
            "main/agent/chat/response_builder.py",
            "main/agent/chat/streaming_handler.py",
            "main/agent/chat/typing_simulator.py",
            
            # Химия
            "main/agent/chem/chemical_classifier.py",
            "main/agent/chem/chemical_service.py",
            "main/agent/chem/chemical_utils.py",
            "main/agent/chem/chemistry_utils.py",
            "main/agent/chem/molecular_properties.py",
            
            # Химический анализ
            "main/agent/chem/analysis/admet_predictor.py",
            "main/agent/chem/analysis/classifier.py",
            "main/agent/chem/analysis/descriptor_calculator.py",
            "main/agent/chem/analysis/electronic_properties.py",
            "main/agent/chem/analysis/graph_features.py",
            "main/agent/chem/analysis/molecule_analyzer.py",
            "main/agent/chem/analysis/physicochemical_props.py",
            "main/agent/chem/analysis/properties_calculator.py",
            "main/agent/chem/analysis/similarity_calculator.py",
            "main/agent/chem/analysis/topological_indices.py",
            "main/agent/chem/analysis/toxicity_predictor.py",
            
            # Химическая БД
            "main/agent/chem/db/cache.py",
            "main/agent/chem/db/indexer.py",
            "main/agent/chem/db/models.py",
            "main/agent/chem/db/queries.py",
            "main/agent/chem/db/repository.py",
            
            # Внешние API
            "main/agent/chem/external/api_rate_limiter.py",
            "main/agent/chem/external/chebi_client.py",
            "main/agent/chem/external/chembl_client.py",
            "main/agent/chem/external/drugbank_client.py",
            "main/agent/chem/external/pdb_client.py",
            "main/agent/chem/external/pubchem_client.py",
            "main/agent/chem/external/uniprot_client.py",
            
            # Обработка изображений
            "main/agent/chem/img/chemical_ocr.py",
            "main/agent/chem/img/diagram_extractor.py",
            "main/agent/chem/img/formula_detector.py",
            "main/agent/chem/img/image_preprocessor.py",
            "main/agent/chem/img/img2mol_wrapper.py",
            "main/agent/chem/img/structure_recognizer.py",
            
            # Парсеры
            "main/agent/chem/parser/formula_parser.py",
            "main/agent/chem/parser/inchi_parser.py",
            "main/agent/chem/parser/iupac_parser.py",
            "main/agent/chem/parser/molfile_parser.py",
            "main/agent/chem/parser/smiles_extractor.py",
            "main/agent/chem/parser/smiles_parser.py",
            "main/agent/chem/parser/structure_parser.py",
            
            # Химические утилиты
            "main/agent/chem/util/analyzer.py",
            "main/agent/chem/util/classifier.py",
            "main/agent/chem/util/features.py",
            "main/agent/chem/util/graph_features.py",
            "main/agent/chem/util/service.py",
            "main/agent/chem/util/similarity.py",
            "main/agent/chem/util/utils_chem.py",
            
            # Валидация химии
            "main/agent/chem/validation/chemical_validator.py",
            "main/agent/chem/validation/consistency_checker.py",
            "main/agent/chem/validation/data_quality_checker.py",
            "main/agent/chem/validation/smiles_validator.py",
            "main/agent/chem/validation/standardizer.py",
            
            # Цитирование
            "main/agent/citation/bibliography_generator.py",
            "main/agent/citation/citation_manager.py",
            "main/agent/citation/citation_style.py",
            "main/agent/citation/citation_validator.py",
            "main/agent/citation/link_formatter.py",
            "main/agent/citation/reference_builder.py",
            "main/agent/citation/source_credibility.py",
            "main/agent/citation/source_tracker.py",
            
            # Ядро агента
            "main/agent/core/agent_base.py",
            "main/agent/core/agent_config.py",
            "main/agent/core/agent_factory.py",
            "main/agent/core/agent_registry.py",
            "main/agent/core/agent_state.py",
            "main/agent/core/agent_utils.py",
            
            # База данных
            "main/agent/database/chemical_models.py",
            "main/agent/database/chronobiotics_schema.py",
            "main/agent/database/connection_pool.py",
            "main/agent/database/db_manager.py",
            "main/agent/database/link_models.py",
            "main/agent/database/migration_handler.py",
            "main/agent/database/query_executor.py",
            "main/agent/database/research_models.py",
            
            # KAG
            "main/agent/kag/chronobiotics_kag.py",
            "main/agent/kag/entity_extractor.py",
            "main/agent/kag/graph_builder.py",
            "main/agent/kag/graph_embedder.py",
            "main/agent/kag/graph_querier.py",
            "main/agent/kag/graph_visualizer.py",
            "main/agent/kag/hybrid_kag_retriever.py",
            "main/agent/kag/inference_engine.py",
            "main/agent/kag/kag_service.py",
            "main/agent/kag/kag_utils.py",
            "main/agent/kag/kg_retriever.py",
            "main/agent/kag/knowledge_graph.py",
            "main/agent/kag/path_finder.py",
            "main/agent/kag/relationship_miner.py",
            "main/agent/kag/schema_manager.py",
            
            # Алгоритмы KAG
            "main/agent/kag/algorithms/centrality.py",
            "main/agent/kag/algorithms/clustering.py",
            "main/agent/kag/algorithms/community.py",
            "main/agent/kag/algorithms/matching.py",
            "main/agent/kag/algorithms/propagation.py",
            "main/agent/kag/algorithms/ranking.py",
            "main/agent/kag/algorithms/similarity.py",
            
            # Модели KAG
            "main/agent/kag/models/edge.py",
            "main/agent/kag/models/entity.py",
            "main/agent/kag/models/graph.py",
            "main/agent/kag/models/node.py",
            "main/agent/kag/models/property.py",
            "main/agent/kag/models/relationship.py",
            "main/agent/kag/models/schema.py",
            
            # Запросы KAG
            "main/agent/kag/queries/biological_queries.py",
            "main/agent/kag/queries/chemical_queries.py",
            "main/agent/kag/queries/clinical_queries.py",
            "main/agent/kag/queries/inference_queries.py",
            "main/agent/kag/queries/mechanism_queries.py",
            "main/agent/kag/queries/similarity_queries.py",
            
            # Хранилище KAG
            "main/agent/kag/storage/backup_manager.py",
            "main/agent/kag/storage/graph_db_manager.py",
            "main/agent/kag/storage/graph_loader.py",
            "main/agent/kag/storage/graph_serializer.py",
            "main/agent/kag/storage/neo4j_store.py",
            "main/agent/kag/storage/networkx_store.py",
            
            # Утилиты KAG
            "main/agent/kag/utils/export_utils.py",
            "main/agent/kag/utils/graph_utils.py",
            "main/agent/kag/utils/performance_utils.py",
            "main/agent/kag/utils/query_utils.py",
            "main/agent/kag/utils/validation_utils.py",
            
            # LLM
            "main/agent/llm/api_router.py",
            "main/agent/llm/cache_manager.py",
            "main/agent/llm/context_manager.py",
            "main/agent/llm/cost_tracker.py",
            "main/agent/llm/evaluation_metrics.py",
            "main/agent/llm/fallback_handler.py",
            "main/agent/llm/fine_tuning_manager.py",
            "main/agent/llm/llm_base.py",
            "main/agent/llm/llm_config.py",
            "main/agent/llm/llm_manager.py",
            "main/agent/llm/llm_provider.py",
            "main/agent/llm/llm_service.py",
            "main/agent/llm/llm_utils.py",
            "main/agent/llm/model_adapter.py",
            "main/agent/llm/model_loader.py",
            "main/agent/llm/prompt_engineer.py",
            "main/agent/llm/rate_limiter.py",
            "main/agent/llm/response_parser.py",
            "main/agent/llm/temperature_manager.py",
            "main/agent/llm/token_counter.py",
            
            # Fine tuning
            "main/agent/llm/fine_tuning/bloom_finetune.py",
            "main/agent/llm/fine_tuning/checkpoint_manager.py",
            "main/agent/llm/fine_tuning/data_preparer.py",
            "main/agent/llm/fine_tuning/dataset_manager.py",
            "main/agent/llm/fine_tuning/evaluator.py",
            "main/agent/llm/fine_tuning/fine_tuning_service.py",
            "main/agent/llm/fine_tuning/hyperparameter_tuner.py",
            "main/agent/llm/fine_tuning/lora_adapter.py",
            "main/agent/llm/fine_tuning/trainer.py",
            "main/agent/llm/fine_tuning/training_config.py",
            
            # Модели LLM
            "main/agent/llm/models/anthropic.py",
            "main/agent/llm/models/bloom.py",
            "main/agent/llm/models/cohere.py",
            "main/agent/llm/models/context_window.py",
            "main/agent/llm/models/custom_model.py",
            "main/agent/llm/models/gemini.py",
            "main/agent/llm/models/llama.py",
            "main/agent/llm/models/local_llm.py",
            "main/agent/llm/models/mistral.py",
            "main/agent/llm/models/model_configs.py",
            "main/agent/llm/models/model_parameters.py",
            "main/agent/llm/models/model_registry.py",
            "main/agent/llm/models/openai_gpt.py",
            "main/agent/llm/models/qwen.py",
            
            # Multimodal агенты
            "main/agent/llm/multimodal_agents/chart_analyzer_agent.py",
            "main/agent/llm/multimodal_agents/chemical_image_agent.py",
            "main/agent/llm/multimodal_agents/fusion_utils.py",
            "main/agent/llm/multimodal_agents/image_analyzer.py",
            "main/agent/llm/multimodal_agents/image_to_smiles_agent.py",
            "main/agent/llm/multimodal_agents/multimodal_agent_base.py",
            "main/agent/llm/multimodal_agents/multimodal_config.py",
            "main/agent/llm/multimodal_agents/multimodal_fusion_agent.py",
            "main/agent/llm/multimodal_agents/multimodal_reasoning_agent.py",
            "main/agent/llm/multimodal_agents/ocr_agent.py",
            "main/agent/llm/multimodal_agents/structure_recognizer.py",
            "main/agent/llm/multimodal_agents/table_extractor_agent.py",
            "main/agent/llm/multimodal_agents/vision_agent.py",
            "main/agent/llm/multimodal_agents/vision_utils.py",
            
            # Multimodal embeddings
            "main/agent/llm/multimodal_agents/embeddings/alignment_module.py",
            "main/agent/llm/multimodal_agents/embeddings/chemical_embedder.py",
            "main/agent/llm/multimodal_agents/embeddings/embedding_fusion.py",
            "main/agent/llm/multimodal_agents/embeddings/image_embedder.py",
            "main/agent/llm/multimodal_agents/embeddings/multimodal_embedder.py",
            "main/agent/llm/multimodal_agents/embeddings/text_embedder.py",
            
            # Multimodal fusion
            "main/agent/llm/multimodal_agents/fusion/attention_fusion.py",
            "main/agent/llm/multimodal_agents/fusion/cross_modal_attention.py",
            "main/agent/llm/multimodal_agents/fusion/early_fusion.py",
            "main/agent/llm/multimodal_agents/fusion/feature_fusion.py",
            "main/agent/llm/multimodal_agents/fusion/hybrid_fusion.py",
            "main/agent/llm/multimodal_agents/fusion/late_fusion.py",
            
            # Vision
            "main/agent/llm/multimodal_agents/vision/chemical_structure_detector.py",
            "main/agent/llm/multimodal_agents/vision/chart_analyzer.py",
            "main/agent/llm/multimodal_agents/vision/formula_recognizer.py",
            "main/agent/llm/multimodal_agents/vision/image_enhancer.py",
            "main/agent/llm/multimodal_agents/vision/image_processor.py",
            "main/agent/llm/multimodal_agents/vision/molecular_diagram_recognizer.py",
            "main/agent/llm/multimodal_agents/vision/table_extractor.py",
            
            # Multimodal LLM
            "main/agent/llm/multimodal_llm/claude_vision.py",
            "main/agent/llm/multimodal_llm/gemini_vision.py",
            "main/agent/llm/multimodal_llm/gpt4_vision.py",
            "main/agent/llm/multimodal_llm/image_processor.py",
            "main/agent/llm/multimodal_llm/llava.py",
            "main/agent/llm/multimodal_llm/multimodal_base.py",
            "main/agent/llm/multimodal_llm/multimodal_prompt.py",
            "main/agent/llm/multimodal_llm/multimodal_response.py",
            "main/agent/llm/multimodal_llm/vision_embedder.py",
            "main/agent/llm/multimodal_llm/vision_utils.py",
            
            # Оптимизация LLM
            "main/agent/llm/optimization/cache_strategy.py",
            "main/agent/llm/optimization/cost_optimizer.py",
            "main/agent/llm/optimization/fallback_strategy.py",
            "main/agent/llm/optimization/latency_optimizer.py",
            "main/agent/llm/optimization/load_balancer.py",
            "main/agent/llm/optimization/model_selector.py",
            "main/agent/llm/optimization/performance_monitor.py",
            "main/agent/llm/optimization/quality_optimizer.py",
            
            # Промпты
            "main/agent/llm/prompts/analysis_prompts.py",
            "main/agent/llm/prompts/chat_prompts.py",
            "main/agent/llm/prompts/chemical_prompts.py",
            "main/agent/llm/prompts/few_shot_examples.py",
            "main/agent/llm/prompts/multimodal_prompts.py",
            "main/agent/llm/prompts/prompt_evaluator.py",
            "main/agent/llm/prompts/prompt_manager.py",
            "main/agent/llm/prompts/prompt_optimizer.py",
            "main/agent/llm/prompts/prompt_selector.py",
            "main/agent/llm/prompts/prompt_template_base.py",
            "main/agent/llm/prompts/prompt_templates.py",
            "main/agent/llm/prompts/prompt_variables.py",
            "main/agent/llm/prompts/research_prompts.py",
            
            # Инструменты LLM
            "main/agent/llm/tools/analysis_tools.py",
            "main/agent/llm/tools/chemical_tools.py",
            "main/agent/llm/tools/data_tools.py",
            "main/agent/llm/tools/function_calling.py",
            "main/agent/llm/tools/search_tools.py",
            "main/agent/llm/tools/tool_adapter.py",
            "main/agent/llm/tools/tool_executor.py",
            "main/agent/llm/tools/tool_registry.py",
            "main/agent/llm/tools/tool_validator.py",
            "main/agent/llm/tools/web_tools.py",
            
            # Команды управления
            "main/agent/management/commands/agent_commands/agent_status.py",
            "main/agent/management/commands/agent_commands/clear_agent_cache.py",
            "main/agent/management/commands/agent_commands/list_agents.py",
            "main/agent/management/commands/agent_commands/reset_agent_state.py",
            "main/agent/management/commands/agent_commands/run_agent_task.py",
            "main/agent/management/commands/agent_commands/start_agent.py",
            "main/agent/management/commands/agent_commands/stop_agent.py",
            
            "main/agent/management/commands/data_commands/backup_database.py",
            "main/agent/management/commands/data_commands/export_chemical_data.py",
            "main/agent/management/commands/data_commands/import_chemical_data.py",
            "main/agent/management/commands/data_commands/restore_database.py",
            "main/agent/management/commands/data_commands/sync_external_apis.py",
            "main/agent/management/commands/data_commands/update_pubchem_data.py",
            "main/agent/management/commands/data_commands/validate_data.py",
            
            "main/agent/management/commands/kag_commands/build_knowledge_graph.py",
            "main/agent/management/commands/kag_commands/export_kg.py",
            "main/agent/management/commands/kag_commands/import_kg.py",
            "main/agent/management/commands/kag_commands/kg_stats.py",
            "main/agent/management/commands/kag_commands/query_kg.py",
            "main/agent/management/commands/kag_commands/update_kg.py",
            "main/agent/management/commands/kag_commands/visualize_kg.py",
            
            "main/agent/management/commands/llm_commands/clear_llm_cache.py",
            "main/agent/management/commands/llm_commands/evaluate_model.py",
            "main/agent/management/commands/llm_commands/fine_tune_model.py",
            "main/agent/management/commands/llm_commands/list_models.py",
            "main/agent/management/commands/llm_commands/llm_stats.py",
            "main/agent/management/commands/llm_commands/switch_model.py",
            "main/agent/management/commands/llm_commands/test_llm.py",
            
            "main/agent/management/commands/rag_commands/build_index.py",
            "main/agent/management/commands/rag_commands/cleanup_index.py",
            "main/agent/management/commands/rag_commands/index_status.py",
            "main/agent/management/commands/rag_commands/optimize_index.py",
            "main/agent/management/commands/rag_commands/rebuild_index.py",
            "main/agent/management/commands/rag_commands/search_index.py",
            "main/agent/management/commands/rag_commands/update_index.py",
            
            "main/agent/management/commands/system_commands/check_dependencies.py",
            "main/agent/management/commands/system_commands/cleanup_system.py",
            "main/agent/management/commands/system_commands/setup_environment.py",
            "main/agent/management/commands/system_commands/system_status.py",
            
            # Память
            "main/agent/memory/cache_manager.py",
            "main/agent/memory/context_memory.py",
            "main/agent/memory/conversation_memory.py",
            "main/agent/memory/knowledge_memory.py",
            "main/agent/memory/memory_consolidation.py",
            "main/agent/memory/memory_indexer.py",
            "main/agent/memory/memory_retriever.py",
            
            # Параллельная обработка
            "main/agent/parallel/dependency_resolver.py",
            "main/agent/parallel/load_balancer.py",
            "main/agent/parallel/parallel_manager.py",
            "main/agent/parallel/progress_tracker.py",
            "main/agent/parallel/result_aggregator.py",
            "main/agent/parallel/task_dispatcher.py",
            "main/agent/parallel/timeout_manager.py",
            "main/agent/parallel/worker_pool.py",
            
            # RAG
            "main/agent/rag/index_builder.py",
            "main/agent/rag/rag_manager.py",
            "main/agent/rag/rag_service.py",
            "main/agent/rag/reranker.py",
            "main/agent/rag/utils_rag.py",
            
            # Чанкинг RAG
            "main/agent/rag/chunking/adaptive_chunker.py",
            "main/agent/rag/chunking/chemical_chunker.py",
            "main/agent/rag/chunking/chunker_base.py",
            "main/agent/rag/chunking/metadata_extractor.py",
            "main/agent/rag/chunking/overlap_strategy.py",
            "main/agent/rag/chunking/semantic_chunker.py",
            "main/agent/rag/chunking/size_optimizer.py",
            "main/agent/rag/chunking/text_chunker.py",
            
            # Эмбеддинги RAG
            "main/agent/rag/embeddings/chemical_embedder.py",
            "main/agent/rag/embeddings/embedding_base.py",
            "main/agent/rag/embeddings/embedding_cache.py",
            "main/agent/rag/embeddings/embedding_manager.py",
            "main/agent/rag/embeddings/mol2vec_embedder.py",
            "main/agent/rag/embeddings/multimodal_embedder.py",
            "main/agent/rag/embeddings/rdkit_fingerprints.py",
            "main/agent/rag/embeddings/sentence_transformer.py",
            "main/agent/rag/embeddings/text_embedder.py",
            "main/agent/rag/embeddings/transformer_embedder.py",
            
            # База знаний RAG
            "main/agent/rag/knowledge_base/document_processor.py",
            "main/agent/rag/knowledge_base/entity_linking.py",
            "main/agent/rag/knowledge_base/fact_extractor.py",
            "main/agent/rag/knowledge_base/knowledge_graph.py",
            "main/agent/rag/knowledge_base/knowledge_manager.py",
            "main/agent/rag/knowledge_base/relationship_miner.py",
            
            # Ранжирование RAG
            "main/agent/rag/reranking/bm25_reranker.py",
            "main/agent/rag/reranking/cross_encoder_reranker.py",
            "main/agent/rag/reranking/diversity_reranker.py",
            "main/agent/rag/reranking/ensemble_reranker.py",
            "main/agent/rag/reranking/relevance_reranker.py",
            "main/agent/rag/reranking/reranker_base.py",
            "main/agent/rag/reranking/similarity_reranker.py",
            
            # Извлечение RAG
            "main/agent/rag/retrievers/chemical_retriever.py",
            "main/agent/rag/retrievers/dense_retriever.py",
            "main/agent/rag/retrievers/ensemble_retriever.py",
            "main/agent/rag/retrievers/hybrid_retriever.py",
            "main/agent/rag/retrievers/keyword_retriever.py",
            "main/agent/rag/retrievers/multimodal_retriever.py",
            "main/agent/rag/retrievers/retriever_base.py",
            "main/agent/rag/retrievers/semantic_retriever.py",
            "main/agent/rag/retrievers/sparse_retriever.py",
            "main/agent/rag/retrievers/text_retriever.py",
            
            # Векторное хранилище
            "main/agent/rag/vector_store/chroma_store.py",
            "main/agent/rag/vector_store/faiss_store.py",
            "main/agent/rag/vector_store/index_manager.py",
            "main/agent/rag/vector_store/pinecone_store.py",
            "main/agent/rag/vector_store/qdrant_store.py",
            "main/agent/rag/vector_store/similarity_search.py",
            "main/agent/rag/vector_store/vector_store_base.py",
            "main/agent/rag/vector_store/weaviate_store.py",
            
            # Ответы
            "main/agent/response/answer_formatter.py",
            "main/agent/response/chat_formatter.py",
            "main/agent/response/citation_integrator.py",
            "main/agent/response/confidence_calculator.py",
            "main/agent/response/markdown_generator.py",
            "main/agent/response/response_builder.py",
            "main/agent/response/response_validator.py",
            "main/agent/response/source_attributor.py",
            
            # Поиск
            "main/agent/search/chronobiotics_query.py",
            "main/agent/search/database_client.py",
            "main/agent/search/link_extractor.py",
            "main/agent/search/query_builder.py",
            "main/agent/search/relevance_scorer.py",
            "main/agent/search/result_fetcher.py",
            "main/agent/search/search_cache.py",
            "main/agent/search/search_engine.py",
            
            # Задачи
            "main/agent/tasks/celery.py",
            "main/agent/tasks/celery_app.py",
            "main/agent/tasks/celery_config.py",
            
            # Задачи агентов
            "main/agent/tasks/agent_tasks/agent_monitoring.py",
            "main/agent/tasks/agent_tasks/analysis_execution.py",
            "main/agent/tasks/agent_tasks/chat_processing.py",
            "main/agent/tasks/agent_tasks/citation_processing.py",
            "main/agent/tasks/agent_tasks/data_collection.py",
            "main/agent/tasks/agent_tasks/parallel_search.py",
            "main/agent/tasks/agent_tasks/report_generation.py",
            "main/agent/tasks/agent_tasks/result_aggregation.py",
            
            # Фоновые задачи
            "main/agent/tasks/background_tasks/batch_analysis.py",
            "main/agent/tasks/background_tasks/cleanup_operations.py",
            "main/agent/tasks/background_tasks/data_processing.py",
            "main/agent/tasks/background_tasks/email_processing.py",
            "main/agent/tasks/background_tasks/file_handling.py",
            "main/agent/tasks/background_tasks/notification_sending.py",
            "main/agent/tasks/background_tasks/report_delivery.py",
            "main/agent/tasks/background_tasks/web_scraping.py",
            
            # Химические задачи
            "main/agent/tasks/chem_tasks/batch_processing.py",
            "main/agent/tasks/chem_tasks/chemical_analysis.py",
            "main/agent/tasks/chem_tasks/data_validation.py",
            "main/agent/tasks/chem_tasks/molecule_processing.py",
            "main/agent/tasks/chem_tasks/property_prediction.py",
            "main/agent/tasks/chem_tasks/similarity_calculation.py",
            "main/agent/tasks/chem_tasks/toxicity_assessment.py",
            
            # KAG задачи
            "main/agent/tasks/kag_tasks/entity_extraction.py",
            "main/agent/tasks/kag_tasks/graph_building.py",
            "main/agent/tasks/kag_tasks/graph_embedding.py",
            "main/agent/tasks/kag_tasks/graph_maintenance.py",
            "main/agent/tasks/kag_tasks/inference_processing.py",
            "main/agent/tasks/kag_tasks/kg_query_processing.py",
            "main/agent/tasks/kag_tasks/relationship_mining.py",
            "main/agent/tasks/kag_tasks/visualization_generation.py",
            
            # LLM задачи
            "main/agent/tasks/llm_tasks/cache_management.py",
            "main/agent/tasks/llm_tasks/cost_calculation.py",
            "main/agent/tasks/llm_tasks/fine_tuning.py",
            "main/agent/tasks/llm_tasks/llm_inference.py",
            "main/agent/tasks/llm_tasks/model_evaluation.py",
            "main/agent/tasks/llm_tasks/performance_testing.py",
            "main/agent/tasks/llm_tasks/prompt_engineering.py",
            "main/agent/tasks/llm_tasks/response_processing.py",
            
            # Периодические задачи
            "main/agent/tasks/periodic_tasks/backup_creation.py",
            "main/agent/tasks/periodic_tasks/cache_cleanup.py",
            "main/agent/tasks/periodic_tasks/data_sync.py",
            "main/agent/tasks/periodic_tasks/graph_refresh.py",
            "main/agent/tasks/periodic_tasks/health_check.py",
            "main/agent/tasks/periodic_tasks/index_update.py",
            "main/agent/tasks/periodic_tasks/maintenance_tasks.py",
            "main/agent/tasks/periodic_tasks/model_retraining.py",
            "main/agent/tasks/periodic_tasks/performance_reporting.py",
            
            # RAG задачи
            "main/agent/tasks/rag_tasks/cache_updating.py",
            "main/agent/tasks/rag_tasks/embedding_generation.py",
            "main/agent/tasks/rag_tasks/index_maintenance.py",
            "main/agent/tasks/rag_tasks/indexing.py",
            "main/agent/tasks/rag_tasks/knowledge_base_update.py",
            "main/agent/tasks/rag_tasks/vector_search.py",
            
            # Утилиты агента
            "main/agent/utils/data_utils.py",
            "main/agent/utils/error_utils.py",
            "main/agent/utils/file_utils.py",
            "main/agent/utils/logging_utils.py",
            "main/agent/utils/security_utils.py",
            "main/agent/utils/text_utils.py",
            "main/agent/utils/time_utils.py",
            "main/agent/utils/validation_utils.py",
            
            # Веб
            "main/agent/web/cache_manager.py",
            "main/agent/web/content_parser.py",
            "main/agent/web/link_crawler.py",
            "main/agent/web/rate_limiter.py",
            "main/agent/web/robots_checker.py",
            "main/agent/web/scraper_engine.py",
            "main/agent/web/web_client.py",
            "main/agent/web/web_content_validator.py",
        ]
        
        for file_path in agent_files:
            self.create_file(file_path, "# Файл агента\n\n")
    
    def _create_api_files(self):
        """Создает файлы API"""
        api_files = [
            "main/api/authentication.py",
            "main/api/docs.py",
            "main/api/filters.py",
            "main/api/pagination.py",
            "main/api/permissions.py",
            "main/api/schemas.py",
            "main/api/serializers.py",
            "main/api/tests.py",
            "main/api/throttling.py",
            "main/api/urls.py",
            "main/api/views.py",
            
            "main/api/v1/serializers.py",
            "main/api/v1/tests.py",
            "main/api/v1/urls.py",
            "main/api/v1/views.py",
            
            "main/api/v1/agents/serializers.py",
            "main/api/v1/agents/urls.py",
            "main/api/v1/agents/views.py",
            
            "main/api/v1/chat/serializers.py",
            "main/api/v1/chat/urls.py",
            "main/api/v1/chat/views.py",
            
            "main/api/v1/chemical/serializers.py",
            "main/api/v1/chemical/urls.py",
            "main/api/v1/chemical/views.py",
            
            "main/api/v1/data/serializers.py",
            "main/api/v1/data/urls.py",
            "main/api/v1/data/views.py",
            
            "main/api/v1/kag/serializers.py",
            "main/api/v1/kag/urls.py",
            "main/api/v1/kag/views.py",
            
            "main/api/v1/rag/serializers.py",
            "main/api/v1/rag/urls.py",
            "main/api/v1/rag/views.py",
            
            "main/api/websocket/consumers.py",
            "main/api/websocket/middleware.py",
            "main/api/websocket/routing.py",
        ]
        
        for file_path in api_files:
            self.create_file(file_path, "# API файл\n\n")
    
    def _create_migration_files(self):
        """Создает файлы миграций"""
        migration_files = [
            "main/migrations/0001_initial.py",
            "main/migrations/0002_chemical_data.py",
            "main/migrations/0002_targets_targetsfullname.py",
            "main/migrations/0003_agent_system.py",
            "main/migrations/0003_alter_targets_targetsfullname.py",
            "main/migrations/0004_alter_chronobiotic_description.py",
            "main/migrations/0004_conversations.py",
            "main/migrations/0005_citations.py",
            "main/migrations/0005_effect_alter_chronobiotic_mechanisms_and_more.py",
            "main/migrations/0006_alter_chronobiotic_effect.py",
            "main/migrations/0006_knowledge_graph.py",
            "main/migrations/0007_articles_remove_chronobiotic_article_and_more.py",
            "main/migrations/0008_remove_chronobiotic_articles_and_more.py",
            "main/migrations/0009_articles_effect_remove_chronobiotic_article_and_more.py",
        ]
        
        for file_path in migration_files:
            self.create_file(file_path, "# Django миграция\n\n")
    
    def _create_static_files(self):
        """Создает статические файлы"""
        # CSS файлы
        css_files = [
            "main/static/css/agent-chat.css",
            "main/static/css/chat-interface.css",
            "main/static/css/citation-styles.css",
            "main/static/css/loading-animations.css",
            "main/static/css/message-bubble.css",
            "main/static/css/responsive-chat.css",
        ]
        
        # JS файлы
        js_files = [
            "main/static/js/agent-chat.js",
            "main/static/js/agent-control.js",
            "main/static/js/chat-streaming.js",
            "main/static/js/citation-display.js",
            "main/static/js/file-upload.js",
            "main/static/js/markdown-render.js",
            "main/static/js/message-handler.js",
            "main/static/js/realtime-updates.js",
            "main/static/js/typing-simulator.js",
        ]
        
        # Изображения (пустые файлы)
        image_files = [
            "main/static/images/agent-avatar.png",
            "main/static/images/citation-icon.png",
            "main/static/images/chronobiotics-logo.png",
            "main/static/images/loading-spinner.gif",
            "main/static/images/source-icon.png",
            "main/static/images/user-avatar.png",
        ]
        
        for file_list, comment in [
            (css_files, "/* CSS файл */"),
            (js_files, "// JavaScript файл"),
            (image_files, ""),
        ]:
            for file_path in file_list:
                self.create_file(file_path, comment)
    
    def _create_template_files(self):
        """Создает шаблонные файлы"""
        template_files = [
            "main/templates/main/agent_chat.html",
            "main/templates/main/agent_settings.html",
            "main/templates/main/base_agent.html",
            "main/templates/main/chat_messages.html",
            "main/templates/main/chat_sidebar.html",
            "main/templates/main/citation_display.html",
            "main/templates/main/conversation_history.html",
            "main/templates/main/file_upload.html",
            "main/templates/main/loading_indicator.html",
            "main/templates/main/message_bubble.html",
            "main/templates/main/search_results.html",
            "main/templates/main/source_references.html",
        ]
        
        for file_path in template_files:
            self.create_file(file_path, "<!-- HTML шаблон -->\n\n")
    
    def _create_media_files(self):
        """Создает медиа файлы"""
        # Фикстуры
        fixture_files = [
            "fixtures/chemical_data.json",
            "fixtures/test_articles.json",
            "fixtures/test_users.json",
        ]
        
        for file_path in fixture_files:
            self.create_file(file_path, "[]\n")
    
    def _create_requirements_files(self):
        """Создает файлы requirements"""
        requirements_files = [
            "requirements/requirements-chem.txt",
            "requirements/requirements-dev.txt",
            "requirements/requirements-kag.txt",
            "requirements/requirements-llm.txt",
            "requirements/requirements-prod.txt",
            "requirements/requirements-rag.txt",
            "requirements/requirements-test.txt",
            "requirements/requirements-vision.txt",
            "requirements/requirements.txt",
        ]
        
        for file_path in requirements_files:
            self.create_file(file_path, "# Requirements\n\n")
    
    def _create_test_files(self):
        """Создает тестовые файлы"""
        test_files = [
            "tests/conftest.py",
            "tests/pytest.ini",
            
            # Бенчмарки
            "tests/benchmarks/benchmark_agents.py",
            "tests/benchmarks/benchmark_chem.py",
            "tests/benchmarks/benchmark_integration.py",
            "tests/benchmarks/benchmark_kag.py",
            "tests/benchmarks/benchmark_llm.py",
            "tests/benchmarks/benchmark_rag.py",
            
            # Фикстуры
            "tests/fixtures/agent_configs.json",
            "tests/fixtures/chemical_data.json",
            "tests/fixtures/kg_data.json",
            "tests/fixtures/llm_responses.json",
            "tests/fixtures/rag_documents.json",
            "tests/fixtures/test_articles.json",
            
            # Тесты агентов
            "tests/test_agent/test_agent_core.py",
            "tests/test_agent/test_agent_integration.py",
            "tests/test_agent/test_chat_agent.py",
            "tests/test_agent/test_chronobiotics_agent.py",
            "tests/test_agent/test_citation_system.py",
            "tests/test_agent/test_parallel_execution.py",
            "tests/test_agent/test_response_formatter.py",
            "tests/test_agent/test_search_agents.py",
            
            # Тесты химии
            "tests/test_chem/test_chemical_integration.py",
            "tests/test_chem/test_chemical_parser.py",
            "tests/test_chem/test_chemical_service.py",
            "tests/test_chem/test_img2mol.py",
            "tests/test_chem/test_molecule_analyzer.py",
            "tests/test_chem/test_properties_calculator.py",
            "tests/test_chem/test_similarity_calculator.py",
            
            # Тесты KAG
            "tests/test_kag/test_entity_extraction.py",
            "tests/test_kag/test_graph_builder.py",
            "tests/test_kag/test_graph_query.py",
            "tests/test_kag/test_inference_engine.py",
            "tests/test_kag/test_kag_integration.py",
            "tests/test_kag/test_kg_retriever.py",
            "tests/test_kag/test_knowledge_graph.py",
            
            # Тесты LLM
            "tests/test_llm/test_fine_tuning.py",
            "tests/test_llm/test_llm_integration.py",
            "tests/test_llm/test_llm_models.py",
            "tests/test_llm/test_multimodal_agents.py",
            "tests/test_llm/test_multimodal_llm.py",
            "tests/test_llm/test_prompt_engineering.py",
            "tests/test_llm/test_tools.py",
            
            # Тесты RAG
            "tests/test_rag/test_chunking.py",
            "tests/test_rag/test_embedding.py",
            "tests/test_rag/test_index_builder.py",
            "tests/test_rag/test_rag_integration.py",
            "tests/test_rag/test_reranker.py",
            "tests/test_rag/test_retriever.py",
            "tests/test_rag/test_vector_store.py",
            
            # Тесты задач
            "tests/test_tasks/test_agent_tasks.py",
            "tests/test_tasks/test_chem_tasks.py",
            "tests/test_tasks/test_kag_tasks.py",
            "tests/test_tasks/test_llm_tasks.py",
            "tests/test_tasks/test_periodic_tasks.py",
            "tests/test_tasks/test_rag_tasks.py",
            
            # Тесты утилит
            "tests/test_utils/test_data_processing.py",
            "tests/test_utils/test_error_handling.py",
            "tests/test_utils/test_file_handling.py",
            "tests/test_utils/test_logging.py",
            "tests/test_utils/test_monitoring.py",
            "tests/test_utils/test_security.py",
            "tests/test_utils/test_time_utils.py",
        ]
        
        for file_path in test_files:
            if file_path.endswith('.py'):
                self.create_file(file_path, "# Тест\n\n")
            elif file_path.endswith('.json'):
                self.create_file(file_path, "[]\n")
    
    def _create_utils_files(self):
        """Создает файлы утилит"""
        utils_files = [
            # Основные утилиты
            "utils/article_analyzer.py",
            "utils/chemical_classifier.py",
            "utils/chemistry_utils.py",
            "utils/converters.py",
            "utils/data_utils.py",
            "utils/decorators.py",
            "utils/error_utils.py",
            "utils/file_cache.py",
            "utils/file_utils.py",
            "utils/helpers.py",
            "utils/logger.py",
            "utils/logging_utils.py",
            "utils/model_adapters.py",
            "utils/molecular_properties.py",
            "utils/security_utils.py",
            "utils/text_utils.py",
            "utils/time_utils.py",
            "utils/utils.py",
            "utils/validation_utils.py",
            "utils/validators.py",
            
            # Обработка данных
            "utils/data_processing/data_cleaner.py",
            "utils/data_processing/data_filter.py",
            "utils/data_processing/data_formatter.py",
            "utils/data_processing/data_normalizer.py",
            "utils/data_processing/data_quality.py",
            "utils/data_processing/data_serializer.py",
            "utils/data_processing/data_transformer.py",
            "utils/data_processing/data_validator.py",
            
            # Обработка ошибок
            "utils/error_handling/circuit_breaker.py",
            "utils/error_handling/error_codes.py",
            "utils/error_handling/error_recovery.py",
            "utils/error_handling/error_reporter.py",
            "utils/error_handling/exception_handler.py",
            "utils/error_handling/fallback_handler.py",
            "utils/error_handling/graceful_degradation.py",
            "utils/error_handling/retry_manager.py",
            
            # Работа с файлами
            "utils/file_handling/archive_handler.py",
            "utils/file_handling/backup_manager.py",
            "utils/file_handling/downloader.py",
            "utils/file_handling/file_converter.py",
            "utils/file_handling/file_manager.py",
            "utils/file_handling/file_storage.py",
            "utils/file_handling/file_validator.py",
            "utils/file_handling/uploader.py",
            
            # Логирование
            "utils/logging/audit_logger.py",
            "utils/logging/error_logger.py",
            "utils/logging/log_analyzer.py",
            "utils/logging/log_formatter.py",
            "utils/logging/log_handler.py",
            "utils/logging/log_rotation.py",
            "utils/logging/metrics_logger.py",
            "utils/logging/performance_logger.py",
            
            # Мониторинг
            "utils/monitoring/alert_manager.py",
            "utils/monitoring/dashboard_generator.py",
            "utils/monitoring/health_check.py",
            "utils/monitoring/metrics.py",
            "utils/monitoring/performance_monitor.py",
            "utils/monitoring/resource_tracker.py",
            "utils/monitoring/system_monitor.py",
            "utils/monitoring/tracing.py",
            "utils/monitoring/usage_tracker.py",
            
            # Сеть
            "utils/network/api_client.py",
            "utils/network/connection_pool.py",
            "utils/network/dns_resolver.py",
            "utils/network/http_client.py",
            "utils/network/network_monitor.py",
            "utils/network/proxy_manager.py",
            "utils/network/websocket_client.py",
            
            # Безопасность
            "utils/security/access_control.py",
            "utils/security/authentication.py",
            "utils/security/authorization.py",
            "utils/security/encryption.py",
            "utils/security/input_validator.py",
            "utils/security/rate_limiter.py",
            "utils/security/sanitizer.py",
            "utils/security/security_audit.py",
            "utils/security/token_manager.py",
            
            # Время
            "utils/time/cache_expiry.py",
            "utils/time/cron_parser.py",
            "utils/time/date_parser.py",
            "utils/time/rate_limiter.py",
            "utils/time/scheduler.py",
            "utils/time/time_utils.py",
            "utils/time/timezone_handler.py",
        ]
        
        for file_path in utils_files:
            self.create_file(file_path, "# Утилита\n\n")
    
    def _create_root_files(self):
        """Создает корневые файлы проекта"""
        # .env.example
        env_content = """# Конфигурация окружения Chronobiotic
# Копируйте этот файл в .env и настройте значения

# Django настройки
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# База данных
DATABASE_URL=sqlite:///db.sqlite3

# API ключи
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Настройки LLM
DEFAULT_LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002

# Настройки RAG
VECTOR_STORE_TYPE=chroma

# Настройки KAG
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Настройки Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Логирование
LOG_LEVEL=INFO
LOG_FILE=./logs/chronobiotic.log
"""
        
        # .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Django
*.log
*.pot
*.pyc
__pycache__/
db.sqlite3
media/
staticfiles/

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
chroma_db/
logs/
*.db
*.sqlite3
*.dump
db.sqlite3
media/
staticfiles/

# Environment variables
.env
.env.local
.env.*.local

# Jupyter Notebook
.ipynb_checkpoints

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/
.coverage
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Docker
*.dockerignore
Dockerfile

# Backups
*.bak
*.backup
"""
        
        # manage.py
        manage_content = """#!/usr/bin/env python
import os
import sys

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
"""
        
        # README.md
        readme_content = """# Chronobiotic

Исследовательская система для анализа хронобиотиков с использованием AI-агентов.

## Описание проекта

Chronobiotic - это комплексная система для исследования и анализа хронобиотиков (веществ, влияющих на циркадные ритмы).

## Установка

1. Клонируйте репозиторий
2. Создайте виртуальное окружение
3. Установите зависимости: `pip install -r requirements.txt`
4. Настройте переменные окружения: `cp .env.example .env`
5. Выполните миграции: `python manage.py migrate`
6. Запустите сервер: `python manage.py runserver`

## Структура проекта

Проект организован по модульному принципу с четким разделением ответственности.
"""
        
        # docker-compose.yml
        docker_compose_content = """version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: chronobiotic
      POSTGRES_USER: chronobiotic
      POSTGRES_PASSWORD: chronobiotic_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/chronobiotic_pass
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgres://chronobiotic:chronobiotic_pass@db:5432/chronobiotic
      REDIS_URL: redis://redis:6379/0
      NEO4J_URL: bolt://neo4j:7687
    depends_on:
      - db
      - redis
      - neo4j

volumes:
  postgres_data:
  neo4j_data:
  neo4j_logs:
"""
        
        # dockerfile
        dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
COPY requirements/ requirements/

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY . .

# Создание пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Порт для Django
EXPOSE 8000

# Команда по умолчанию
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
"""
        
        # load_data.py
        load_data_content = """#!/usr/bin/env python
\"\"\"Скрипт для загрузки данных\"\"\"
import os
import sys
import django
import json
from pathlib import Path

# Настройка Django
sys.path.append(str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def main():
    \"\"\"Основная функция\"\"\"
    print("Загрузка данных для Chronobiotic...")

if __name__ == "__main__":
    main()
"""
        
        # Создаем корневые файлы
        files_content = {
            ".env.example": env_content,
            ".gitignore": gitignore_content,
            "manage.py": manage_content,
            "README.md": readme_content,
            "docker-compose.yml": docker_compose_content,
            "dockerfile": dockerfile_content,
            "load_data.py": load_data_content,
            "requirements.txt": "# Основные зависимости\nDjango>=4.2\n",
            "db.sqlite3": "",
            "db0311.json": "{}",
        }
        
        for file_path, content in files_content.items():
            self.create_file(file_path, content)


def main():
    """Основная функция"""
    creator = ProjectCreator()
    creator.create_structure()


if __name__ == "__main__":
    main()
