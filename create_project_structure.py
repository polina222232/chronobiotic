import os
import sys


def create_project_structure(base_path=""):
    """Создает всю структуру проекта с пустыми файлами"""
    
    # Основные директории
    directories = [
        # Корневые директории
        "chronobiotic",
        "fixtures",
        "main",
        "media/chemical_structures",
        "media/documents",
        "media/user_uploads",
        "requirements",
        "tests",
        "utils",
        
        # chronobiotic/
        "chronobiotic",
        
        # main/
        "main/__pycache__",  # Для игнорирования
        "main/agent",
        "main/agent/agents",
        "main/agent/agents/analysis_agents",
        "main/agent/agents/assistant_agents",
        "main/agent/agents/chronobiotics_agents",
        "main/agent/agents/citation_agents",
        "main/agent/agents/data_agents",
        "main/agent/agents/multilingual_agents",
        "main/agent/agents/research_agents",
        "main/agent/agents/voice_agents",
        "main/agent/analysis",
        "main/agent/audio",
        "main/agent/chat",
        "main/agent/chem",
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
        "main/agent/geo",
        "main/agent/kag",
        "main/agent/kag/algorithms",
        "main/agent/kag/models",
        "main/agent/kag/queries",
        "main/agent/kag/storage",
        "main/agent/kag/utils",
        "main/agent/language_models",
        "main/agent/llm",
        "main/agent/llm/fine_tuning",
        "main/agent/llm/models",
        "main/agent/llm/multimodal_agents",
        "main/agent/llm/multimodal_agents/embeddings",
        "main/agent/llm/multimodal_agents/fusion",
        "main/agent/llm/multimodal_agents/vision",
        "main/agent/llm/multimodal_llm",
        "main/agent/llm/optimization",
        "main/agent/llm/prompts",
        "main/agent/llm/tools",
        "main/agent/localization",
        "main/agent/management",
        "main/agent/management/commands",
        "main/agent/management/commands/agent_commands",
        "main/agent/management/commands/data_commands",
        "main/agent/management/commands/kag_commands",
        "main/agent/management/commands/llm_commands",
        "main/agent/management/commands/multilingual_commands",
        "main/agent/management/commands/rag_commands",
        "main/agent/management/commands/system_commands",
        "main/agent/management/commands/voice_commands",
        "main/agent/memory",
        "main/agent/parallel",
        "main/agent/rag",
        "main/agent/rag/chunking",
        "main/agent/rag/embeddings",
        "main/agent/rag/knowledge_base",
        "main/agent/rag/reranking",
        "main/agent/rag/retrievers",
        "main/agent/rag/vector_store",
        "main/agent/response",
        "main/agent/search",
        "main/agent/stt",
        "main/agent/tasks",
        "main/agent/tasks/agent_tasks",
        "main/agent/tasks/background_tasks",
        "main/agent/tasks/chem_tasks",
        "main/agent/tasks/kag_tasks",
        "main/agent/tasks/llm_tasks",
        "main/agent/tasks/periodic_tasks",
        "main/agent/tasks/rag_tasks",
        "main/agent/tasks/voice_tasks",
        "main/agent/tts",
        "main/agent/utils",
        "main/agent/voice_ui",
        "main/agent/web",
        "main/api",
        "main/api/v1",
        "main/api/v1/agents",
        "main/api/v1/chat",
        "main/api/v1/chemical",
        "main/api/v1/data",
        "main/api/v1/kag",
        "main/api/v1/multilingual",
        "main/api/v1/rag",
        "main/api/v1/voice",
        "main/api/websocket",
        "main/migrations",
        "main/static/main",
        "main/static/main/css",
        "main/static/main/js",
        "main/static/main/audio",
        "main/static/main/images",
        "main/templates/main",
        
        # tests/
        "tests/benchmarks",
        "tests/fixtures",
        "tests/fixtures/test_audio",
        "tests/fixtures/test_images",
        "tests/test_agent",
        "tests/test_chem",
        "tests/test_kag",
        "tests/test_llm",
        "tests/test_rag",
        "tests/test_tasks",
        "tests/test_utils",
        
        # utils/
        "utils/data_processing",
        "utils/error_handling",
        "utils/file_handling",
        "utils/logging",
        "utils/monitoring",
        "utils/network",
        "utils/security",
        "utils/time",
    ]
    
    # Файлы для создания (пустые или с минимальным содержимым)
    files = [
        # Корневые файлы
        "manage.py",
        "requirements.txt",
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        ".gitignore",
        "pyproject.toml",
        "setup.py",
        
        # chronobiotic/
        "chronobiotic/__init__.py",
        "chronobiotic/asgi.py",
        "chronobiotic/settings.py",
        "chronobiotic/settings_dev.py",
        "chronobiotic/settings_prod.py",
        "chronobiotic/settings_test.py",
        "chronobiotic/urls.py",
        "chronobiotic/wsgi.py",
        
        # fixtures/
        "fixtures/chemical_data.json",
        "fixtures/test_articles.json",
        "fixtures/test_users.json",
        
        # main/ основные файлы
        "main/__init__.py",
        "main/admin.py",
        "main/apps.py",
        "main/models.py",
        "main/tests.py",
        "main/urls.py",
        "main/views.py",
        
        # main/agent/ основные файлы
        "main/agent/__init__.py",
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
        
        # main/agent/agents/
        "main/agent/agents/__init__.py",
        "main/agent/agents/base_agent.py",
        
        # main/agent/agents/analysis_agents/
        "main/agent/agents/analysis_agents/__init__.py",
        "main/agent/agents/analysis_agents/chemical_analyzer.py",
        "main/agent/agents/analysis_agents/efficacy_evaluator.py",
        "main/agent/agents/analysis_agents/interaction_analyzer.py",
        "main/agent/agents/analysis_agents/property_predictor.py",
        "main/agent/agents/analysis_agents/similarity_finder.py",
        "main/agent/agents/analysis_agents/toxicity_estimator.py",
        
        # main/agent/agents/assistant_agents/
        "main/agent/agents/assistant_agents/__init__.py",
        "main/agent/agents/assistant_agents/chat_agent.py",
        "main/agent/agents/assistant_agents/explanation_agent.py",
        "main/agent/agents/assistant_agents/qa_agent.py",
        "main/agent/agents/assistant_agents/recommendation_agent.py",
        "main/agent/agents/assistant_agents/summarizer_agent.py",
        
        # main/agent/agents/chronobiotics_agents/
        "main/agent/agents/chronobiotics_agents/__init__.py",
        "main/agent/agents/chronobiotics_agents/chronobiology_expert.py",
        "main/agent/agents/chronobiotics_agents/chronobiotics_searcher.py",
        "main/agent/agents/chronobiotics_agents/clinical_data_finder.py",
        "main/agent/agents/chronobiotics_agents/literature_miner.py",
        "main/agent/agents/chronobiotics_agents/mechanism_researcher.py",
        "main/agent/agents/chronobiotics_agents/substance_analyzer.py",
        
        # main/agent/agents/citation_agents/
        "main/agent/agents/citation_agents/__init__.py",
        "main/agent/agents/citation_agents/bibliography_builder.py",
        "main/agent/agents/citation_agents/citation_extractor.py",
        "main/agent/agents/citation_agents/reference_formatter.py",
        "main/agent/agents/citation_agents/source_tracker.py",
        "main/agent/agents/citation_agents/source_validator.py",
        
        # main/agent/agents/data_agents/
        "main/agent/agents/data_agents/__init__.py",
        "main/agent/agents/data_agents/content_analyzer.py",
        "main/agent/agents/data_agents/data_storage.py",
        "main/agent/agents/data_agents/data_validator.py",
        "main/agent/agents/data_agents/database_searcher.py",
        "main/agent/agents/data_agents/link_follower.py",
        "main/agent/agents/data_agents/web_scraper.py",
        
        # main/agent/agents/multilingual_agents/
        "main/agent/agents/multilingual_agents/__init__.py",
        "main/agent/agents/multilingual_agents/language_detector_agent.py",
        "main/agent/agents/multilingual_agents/language_identifier.py",
        "main/agent/agents/multilingual_agents/multilingual_chat_agent.py",
        "main/agent/agents/multilingual_agents/translation_agent.py",
        "main/agent/agents/multilingual_agents/localization_agent.py",
        
        # main/agent/agents/research_agents/
        "main/agent/agents/research_agents/__init__.py",
        "main/agent/agents/research_agents/clinical_trial_finder.py",
        "main/agent/agents/research_agents/hypothesis_generator.py",
        "main/agent/agents/research_agents/literature_reviewer.py",
        "main/agent/agents/research_agents/mechanism_investigator.py",
        "main/agent/agents/research_agents/patent_searcher.py",
        
        # main/agent/agents/voice_agents/
        "main/agent/agents/voice_agents/__init__.py",
        "main/agent/agents/voice_agents/speech_recognition_agent.py",
        "main/agent/agents/voice_agents/speech_synthesis_agent.py",
        "main/agent/agents/voice_agents/voice_interface_agent.py",
        "main/agent/agents/voice_agents/audio_processor_agent.py",
        "main/agent/agents/voice_agents/multimodal_voice_agent.py",
        
        # main/agent/analysis/
        "main/agent/analysis/__init__.py",
        "main/agent/analysis/analysis_engine.py",
        "main/agent/analysis/analysis_validator.py",
        "main/agent/analysis/chemical_analyzer.py",
        "main/agent/analysis/chronobiotics_analyzer.py",
        "main/agent/analysis/data_processor.py",
        "main/agent/analysis/insight_extractor.py",
        "main/agent/analysis/pattern_finder.py",
        "main/agent/analysis/text_analyzer.py",
        
        # main/agent/audio/
        "main/agent/audio/__init__.py",
        "main/agent/audio/audio_converter.py",
        "main/agent/audio/audio_enhancer.py",
        "main/agent/audio/audio_streamer.py",
        "main/agent/audio/voice_activity_detector.py",
        "main/agent/audio/wake_word_detector.py",
        
        # main/agent/chat/
        "main/agent/chat/__init__.py",
        "main/agent/chat/chat_engine.py",
        "main/agent/chat/chat_formatter.py",
        "main/agent/chat/chat_history.py",
        "main/agent/chat/conversation_manager.py",
        "main/agent/chat/message_handler.py",
        "main/agent/chat/multilingual_chat_engine.py",
        "main/agent/chat/response_builder.py",
        "main/agent/chat/streaming_handler.py",
        "main/agent/chat/typing_simulator.py",
        "main/agent/chat/voice_chat_engine.py",
        "main/agent/chat/audio_message_handler.py",
        
        # main/agent/chem/
        "main/agent/chem/__init__.py",
        "main/agent/chem/chemical_classifier.py",
        "main/agent/chem/chemical_service.py",
        "main/agent/chem/chemical_utils.py",
        "main/agent/chem/chemistry_utils.py",
        "main/agent/chem/molecular_properties.py",
        
        # main/agent/chem/analysis/
        "main/agent/chem/analysis/__init__.py",
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
        
        # main/agent/chem/db/
        "main/agent/chem/db/__init__.py",
        "main/agent/chem/db/cache.py",
        "main/agent/chem/db/indexer.py",
        "main/agent/chem/db/models.py",
        "main/agent/chem/db/queries.py",
        "main/agent/chem/db/repository.py",
        
        # main/agent/chem/external/
        "main/agent/chem/external/__init__.py",
        "main/agent/chem/external/api_rate_limiter.py",
        "main/agent/chem/external/chebi_client.py",
        "main/agent/chem/external/chembl_client.py",
        "main/agent/chem/external/drugbank_client.py",
        "main/agent/chem/external/pdb_client.py",
        "main/agent/chem/external/pubchem_client.py",
        "main/agent/chem/external/uniprot_client.py",
        
        # main/agent/chem/img/
        "main/agent/chem/img/__init__.py",
        "main/agent/chem/img/chemical_ocr.py",
        "main/agent/chem/img/diagram_extractor.py",
        "main/agent/chem/img/formula_detector.py",
        "main/agent/chem/img/image_preprocessor.py",
        "main/agent/chem/img/img2mol_wrapper.py",
        "main/agent/chem/img/structure_recognizer.py",
        
        # main/agent/chem/parser/
        "main/agent/chem/parser/__init__.py",
        "main/agent/chem/parser/formula_parser.py",
        "main/agent/chem/parser/inchi_parser.py",
        "main/agent/chem/parser/iupac_parser.py",
        "main/agent/chem/parser/molfile_parser.py",
        "main/agent/chem/parser/smiles_extractor.py",
        "main/agent/chem/parser/smiles_parser.py",
        "main/agent/chem/parser/structure_parser.py",
        
        # main/agent/chem/util/
        "main/agent/chem/util/__init__.py",
        "main/agent/chem/util/analyzer.py",
        "main/agent/chem/util/classifier.py",
        "main/agent/chem/util/features.py",
        "main/agent/chem/util/graph_features.py",
        "main/agent/chem/util/service.py",
        "main/agent/chem/util/similarity.py",
        "main/agent/chem/util/utils_chem.py",
        
        # main/agent/chem/validation/
        "main/agent/chem/validation/__init__.py",
        "main/agent/chem/validation/chemical_validator.py",
        "main/agent/chem/validation/consistency_checker.py",
        "main/agent/chem/validation/data_quality_checker.py",
        "main/agent/chem/validation/smiles_validator.py",
        "main/agent/chem/validation/standardizer.py",
        
        # main/agent/citation/
        "main/agent/citation/__init__.py",
        "main/agent/citation/bibliography_generator.py",
        "main/agent/citation/citation_manager.py",
        "main/agent/citation/citation_style.py",
        "main/agent/citation/citation_validator.py",
        "main/agent/citation/link_formatter.py",
        "main/agent/citation/reference_builder.py",
        "main/agent/citation/source_credibility.py",
        "main/agent/citation/source_tracker.py",
        
        # main/agent/core/
        "main/agent/core/__init__.py",
        "main/agent/core/agent_base.py",
        "main/agent/core/agent_config.py",
        "main/agent/core/agent_factory.py",
        "main/agent/core/agent_registry.py",
        "main/agent/core/agent_state.py",
        "main/agent/core/agent_utils.py",
        "main/agent/core/multilingual_config.py",
        "main/agent/core/voice_config.py",
        
        # main/agent/database/
        "main/agent/database/__init__.py",
        "main/agent/database/chemical_models.py",
        "main/agent/database/chronobiotics_schema.py",
        "main/agent/database/connection_pool.py",
        "main/agent/database/db_manager.py",
        "main/agent/database/link_models.py",
        "main/agent/database/migration_handler.py",
        "main/agent/database/query_executor.py",
        "main/agent/database/research_models.py",
        "main/agent/database/voice_models.py",
        
        # main/agent/geo/
        "main/agent/geo/__init__.py",
        "main/agent/geo/ip_geolocator.py",
        "main/agent/geo/geo_language_mapper.py",
        "main/agent/geo/timezone_manager.py",
        
        # main/agent/kag/
        "main/agent/kag/__init__.py",
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
        
        # main/agent/kag/algorithms/
        "main/agent/kag/algorithms/__init__.py",
        "main/agent/kag/algorithms/centrality.py",
        "main/agent/kag/algorithms/clustering.py",
        "main/agent/kag/algorithms/community.py",
        "main/agent/kag/algorithms/matching.py",
        "main/agent/kag/algorithms/propagation.py",
        "main/agent/kag/algorithms/ranking.py",
        "main/agent/kag/algorithms/similarity.py",
        
        # main/agent/kag/models/
        "main/agent/kag/models/__init__.py",
        "main/agent/kag/models/edge.py",
        "main/agent/kag/models/entity.py",
        "main/agent/kag/models/graph.py",
        "main/agent/kag/models/node.py",
        "main/agent/kag/models/property.py",
        "main/agent/kag/models/relationship.py",
        "main/agent/kag/models/schema.py",
        
        # main/agent/kag/queries/
        "main/agent/kag/queries/__init__.py",
        "main/agent/kag/queries/biological_queries.py",
        "main/agent/kag/queries/chemical_queries.py",
        "main/agent/kag/queries/clinical_queries.py",
        "main/agent/kag/queries/inference_queries.py",
        "main/agent/kag/queries/mechanism_queries.py",
        "main/agent/kag/queries/similarity_queries.py",
        
        # main/agent/kag/storage/
        "main/agent/kag/storage/__init__.py",
        "main/agent/kag/storage/backup_manager.py",
        "main/agent/kag/storage/graph_db_manager.py",
        "main/agent/kag/storage/graph_loader.py",
        "main/agent/kag/storage/graph_serializer.py",
        "main/agent/kag/storage/neo4j_store.py",
        "main/agent/kag/storage/networkx_store.py",
        
        # main/agent/kag/utils/
        "main/agent/kag/utils/__init__.py",
        "main/agent/kag/utils/export_utils.py",
        "main/agent/kag/utils/graph_utils.py",
        "main/agent/kag/utils/performance_utils.py",
        "main/agent/kag/utils/query_utils.py",
        "main/agent/kag/utils/validation_utils.py",
        
        # main/agent/language_models/
        "main/agent/language_models/__init__.py",
        "main/agent/language_models/multilingual_llm.py",
        "main/agent/language_models/language_specific_prompts.py",
        "main/agent/language_models/cross_lingual_embeddings.py",
        
        # main/agent/llm/
        "main/agent/llm/__init__.py",
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
        
        # main/agent/llm/fine_tuning/
        "main/agent/llm/fine_tuning/__init__.py",
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
        
        # main/agent/llm/models/
        "main/agent/llm/models/__init__.py",
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
        "main/agent/llm/models/multilingual_models.py",
        "main/agent/llm/models/openai_gpt.py",
        "main/agent/llm/models/qwen.py",
        "main/agent/llm/models/voice_models.py",
        
        # main/agent/llm/multimodal_agents/
        "main/agent/llm/multimodal_agents/__init__.py",
        "main/agent/llm/multimodal_agents/audio_visual_agent.py",
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
        
        # main/agent/llm/multimodal_agents/embeddings/
        "main/agent/llm/multimodal_agents/embeddings/__init__.py",
        "main/agent/llm/multimodal_agents/embeddings/alignment_module.py",
        "main/agent/llm/multimodal_agents/embeddings/chemical_embedder.py",
        "main/agent/llm/multimodal_agents/embeddings/embedding_fusion.py",
        "main/agent/llm/multimodal_agents/embeddings/image_embedder.py",
        "main/agent/llm/multimodal_agents/embeddings/multimodal_embedder.py",
        "main/agent/llm/multimodal_agents/embeddings/text_embedder.py",
        
        # main/agent/llm/multimodal_agents/fusion/
        "main/agent/llm/multimodal_agents/fusion/__init__.py",
        "main/agent/llm/multimodal_agents/fusion/attention_fusion.py",
        "main/agent/llm/multimodal_agents/fusion/cross_modal_attention.py",
        "main/agent/llm/multimodal_agents/fusion/early_fusion.py",
        "main/agent/llm/multimodal_agents/fusion/feature_fusion.py",
        "main/agent/llm/multimodal_agents/fusion/hybrid_fusion.py",
        "main/agent/llm/multimodal_agents/fusion/late_fusion.py",
        
        # main/agent/llm/multimodal_agents/vision/
        "main/agent/llm/multimodal_agents/vision/__init__.py",
        "main/agent/llm/multimodal_agents/vision/chemical_structure_detector.py",
        "main/agent/llm/multimodal_agents/vision/chart_analyzer.py",
        "main/agent/llm/multimodal_agents/vision/formula_recognizer.py",
        "main/agent/llm/multimodal_agents/vision/image_enhancer.py",
        "main/agent/llm/multimodal_agents/vision/image_processor.py",
        "main/agent/llm/multimodal_agents/vision/molecular_diagram_recognizer.py",
        "main/agent/llm/multimodal_agents/vision/table_extractor.py",
        
        # main/agent/llm/multimodal_llm/
        "main/agent/llm/multimodal_llm/__init__.py",
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
        
        # main/agent/llm/optimization/
        "main/agent/llm/optimization/__init__.py",
        "main/agent/llm/optimization/cache_strategy.py",
        "main/agent/llm/optimization/cost_optimizer.py",
        "main/agent/llm/optimization/fallback_strategy.py",
        "main/agent/llm/optimization/latency_optimizer.py",
        "main/agent/llm/optimization/load_balancer.py",
        "main/agent/llm/optimization/model_selector.py",
        "main/agent/llm/optimization/performance_monitor.py",
        "main/agent/llm/optimization/quality_optimizer.py",
        
        # main/agent/llm/prompts/
        "main/agent/llm/prompts/__init__.py",
        "main/agent/llm/prompts/analysis_prompts.py",
        "main/agent/llm/prompts/chat_prompts.py",
        "main/agent/llm/prompts/chemical_prompts.py",
        "main/agent/llm/prompts/few_shot_examples.py",
        "main/agent/llm/prompts/multilingual_prompts.py",
        "main/agent/llm/prompts/multimodal_prompts.py",
        "main/agent/llm/prompts/prompt_evaluator.py",
        "main/agent/llm/prompts/prompt_manager.py",
        "main/agent/llm/prompts/prompt_optimizer.py",
        "main/agent/llm/prompts/prompt_selector.py",
        "main/agent/llm/prompts/prompt_template_base.py",
        "main/agent/llm/prompts/prompt_templates.py",
        "main/agent/llm/prompts/prompt_variables.py",
        "main/agent/llm/prompts/research_prompts.py",
        
        # main/agent/llm/tools/
        "main/agent/llm/tools/__init__.py",
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
        
        # main/agent/localization/
        "main/agent/localization/__init__.py",
        "main/agent/localization/locale_detector.py",
        "main/agent/localization/message_localizer.py",
        "main/agent/localization/format_localizer.py",
        "main/agent/localization/ui_localizer.py",
        
        # main/agent/management/
        "main/agent/management/__init__.py",
        
        # main/agent/management/commands/
        "main/agent/management/commands/__init__.py",
        
        # main/agent/management/commands/agent_commands/
        "main/agent/management/commands/agent_commands/__init__.py",
        "main/agent/management/commands/agent_commands/agent_status.py",
        "main/agent/management/commands/agent_commands/clear_agent_cache.py",
        "main/agent/management/commands/agent_commands/list_agents.py",
        "main/agent/management/commands/agent_commands/reset_agent_state.py",
        "main/agent/management/commands/agent_commands/run_agent_task.py",
        "main/agent/management/commands/agent_commands/start_agent.py",
        "main/agent/management/commands/agent_commands/stop_agent.py",
        
        # main/agent/management/commands/data_commands/
        "main/agent/management/commands/data_commands/__init__.py",
        "main/agent/management/commands/data_commands/backup_database.py",
        "main/agent/management/commands/data_commands/export_chemical_data.py",
        "main/agent/management/commands/data_commands/import_chemical_data.py",
        "main/agent/management/commands/data_commands/restore_database.py",
        "main/agent/management/commands/data_commands/sync_external_apis.py",
        "main/agent/management/commands/data_commands/update_pubchem_data.py",
        "main/agent/management/commands/data_commands/validate_data.py",
        
        # main/agent/management/commands/kag_commands/
        "main/agent/management/commands/kag_commands/__init__.py",
        "main/agent/management/commands/kag_commands/build_knowledge_graph.py",
        "main/agent/management/commands/kag_commands/export_kg.py",
        "main/agent/management/commands/kag_commands/import_kg.py",
        "main/agent/management/commands/kag_commands/kg_stats.py",
        "main/agent/management/commands/kag_commands/query_kg.py",
        "main/agent/management/commands/kag_commands/update_kg.py",
        "main/agent/management/commands/kag_commands/visualize_kg.py",
        
        # main/agent/management/commands/llm_commands/
        "main/agent/management/commands/llm_commands/__init__.py",
        "main/agent/management/commands/llm_commands/clear_llm_cache.py",
        "main/agent/management/commands/llm_commands/evaluate_model.py",
        "main/agent/management/commands/llm_commands/fine_tune_model.py",
        "main/agent/management/commands/llm_commands/list_models.py",
        "main/agent/management/commands/llm_commands/llm_stats.py",
        "main/agent/management/commands/llm_commands/switch_model.py",
        "main/agent/management/commands/llm_commands/test_llm.py",
        
        # main/agent/management/commands/multilingual_commands/
        "main/agent/management/commands/multilingual_commands/__init__.py",
        "main/agent/management/commands/multilingual_commands/add_language_support.py",
        "main/agent/management/commands/multilingual_commands/extract_translatable_strings.py",
        "main/agent/management/commands/multilingual_commands/generate_language_packs.py",
        "main/agent/management/commands/multilingual_commands/language_statistics.py",
        "main/agent/management/commands/multilingual_commands/translate_content.py",
        "main/agent/management/commands/multilingual_commands/update_geo_database.py",
        
        # main/agent/management/commands/rag_commands/
        "main/agent/management/commands/rag_commands/__init__.py",
        "main/agent/management/commands/rag_commands/build_index.py",
        "main/agent/management/commands/rag_commands/cleanup_index.py",
        "main/agent/management/commands/rag_commands/index_status.py",
        "main/agent/management/commands/rag_commands/optimize_index.py",
        "main/agent/management/commands/rag_commands/rebuild_index.py",
        "main/agent/management/commands/rag_commands/search_index.py",
        "main/agent/management/commands/rag_commands/update_index.py",
        
        # main/agent/management/commands/system_commands/
        "main/agent/management/commands/system_commands/__init__.py",
        "main/agent/management/commands/system_commands/check_dependencies.py",
        "main/agent/management/commands/system_commands/cleanup_system.py",
        "main/agent/management/commands/system_commands/setup_environment.py",
        "main/agent/management/commands/system_commands/system_status.py",
        
        # main/agent/management/commands/voice_commands/
        "main/agent/management/commands/voice_commands/__init__.py",
        "main/agent/management/commands/voice_commands/test_voice_recognition.py",
        "main/agent/management/commands/voice_commands/generate_voice_samples.py",
        "main/agent/management/commands/voice_commands/train_wake_word.py",
        "main/agent/management/commands/voice_commands/voice_system_status.py",
        "main/agent/management/commands/voice_commands/optimize_audio_models.py",
        
        # main/agent/memory/
        "main/agent/memory/__init__.py",
        "main/agent/memory/cache_manager.py",
        "main/agent/memory/context_memory.py",
        "main/agent/memory/conversation_memory.py",
        "main/agent/memory/knowledge_memory.py",
        "main/agent/memory/memory_consolidation.py",
        "main/agent/memory/memory_indexer.py",
        "main/agent/memory/memory_retriever.py",
        
        # main/agent/parallel/
        "main/agent/parallel/__init__.py",
        "main/agent/parallel/dependency_resolver.py",
        "main/agent/parallel/load_balancer.py",
        "main/agent/parallel/parallel_manager.py",
        "main/agent/parallel/progress_tracker.py",
        "main/agent/parallel/result_aggregator.py",
        "main/agent/parallel/task_dispatcher.py",
        "main/agent/parallel/timeout_manager.py",
        "main/agent/parallel/worker_pool.py",
        
        # main/agent/rag/
        "main/agent/rag/__init__.py",
        "main/agent/rag/index_builder.py",
        "main/agent/rag/rag_manager.py",
        "main/agent/rag/rag_service.py",
        "main/agent/rag/reranker.py",
        "main/agent/rag/utils_rag.py",
        
        # main/agent/rag/chunking/
        "main/agent/rag/chunking/__init__.py",
        "main/agent/rag/chunking/adaptive_chunker.py",
        "main/agent/rag/chunking/chemical_chunker.py",
        "main/agent/rag/chunking/chunker_base.py",
        "main/agent/rag/chunking/metadata_extractor.py",
        "main/agent/rag/chunking/overlap_strategy.py",
        "main/agent/rag/chunking/semantic_chunker.py",
        "main/agent/rag/chunking/size_optimizer.py",
        "main/agent/rag/chunking/text_chunker.py",
        
        # main/agent/rag/embeddings/
        "main/agent/rag/embeddings/__init__.py",
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
        
        # main/agent/rag/knowledge_base/
        "main/agent/rag/knowledge_base/__init__.py",
        "main/agent/rag/knowledge_base/document_processor.py",
        "main/agent/rag/knowledge_base/entity_linking.py",
        "main/agent/rag/knowledge_base/fact_extractor.py",
        "main/agent/rag/knowledge_base/knowledge_graph.py",
        "main/agent/rag/knowledge_base/knowledge_manager.py",
        "main/agent/rag/knowledge_base/relationship_miner.py",
        
        # main/agent/rag/reranking/
        "main/agent/rag/reranking/__init__.py",
        "main/agent/rag/reranking/bm25_reranker.py",
        "main/agent/rag/reranking/cross_encoder_reranker.py",
        "main/agent/rag/reranking/diversity_reranker.py",
        "main/agent/rag/reranking/ensemble_reranker.py",
        "main/agent/rag/reranking/relevance_reranker.py",
        "main/agent/rag/reranking/reranker_base.py",
        "main/agent/rag/reranking/similarity_reranker.py",
        
        # main/agent/rag/retrievers/
        "main/agent/rag/retrievers/__init__.py",
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
        
        # main/agent/rag/vector_store/
        "main/agent/rag/vector_store/__init__.py",
        "main/agent/rag/vector_store/chroma_store.py",
        "main/agent/rag/vector_store/faiss_store.py",
        "main/agent/rag/vector_store/index_manager.py",
        "main/agent/rag/vector_store/pinecone_store.py",
        "main/agent/rag/vector_store/qdrant_store.py",
        "main/agent/rag/vector_store/similarity_search.py",
        "main/agent/rag/vector_store/vector_store_base.py",
        "main/agent/rag/vector_store/weaviate_store.py",
        
        # main/agent/response/
        "main/agent/response/__init__.py",
        "main/agent/response/answer_formatter.py",
        "main/agent/response/chat_formatter.py",
        "main/agent/response/citation_integrator.py",
        "main/agent/response/confidence_calculator.py",
        "main/agent/response/markdown_generator.py",
        "main/agent/response/response_builder.py",
        "main/agent/response/response_validator.py",
        "main/agent/response/source_attributor.py",
        
        # main/agent/search/
        "main/agent/search/__init__.py",
        "main/agent/search/chronobiotics_query.py",
        "main/agent/search/database_client.py",
        "main/agent/search/link_extractor.py",
        "main/agent/search/query_builder.py",
        "main/agent/search/relevance_scorer.py",
        "main/agent/search/result_fetcher.py",
        "main/agent/search/search_cache.py",
        "main/agent/search/search_engine.py",
        
        # main/agent/stt/
        "main/agent/stt/__init__.py",
        "main/agent/stt/whisper_engine.py",
        "main/agent/stt/google_stt_engine.py",
        "main/agent/stt/azure_stt_engine.py",
        "main/agent/stt/stt_engine_manager.py",
        "main/agent/stt/multilingual_stt.py",
        
        # main/agent/tasks/
        "main/agent/tasks/__init__.py",
        "main/agent/tasks/celery.py",
        "main/agent/tasks/celery_app.py",
        "main/agent/tasks/celery_config.py",
        
        # main/agent/tasks/agent_tasks/
        "main/agent/tasks/agent_tasks/__init__.py",
        "main/agent/tasks/agent_tasks/agent_monitoring.py",
        "main/agent/tasks/agent_tasks/analysis_execution.py",
        "main/agent/tasks/agent_tasks/chat_processing.py",
        "main/agent/tasks/agent_tasks/citation_processing.py",
        "main/agent/tasks/agent_tasks/data_collection.py",
        "main/agent/tasks/agent_tasks/parallel_search.py",
        "main/agent/tasks/agent_tasks/report_generation.py",
        "main/agent/tasks/agent_tasks/result_aggregation.py",
        
        # main/agent/tasks/background_tasks/
        "main/agent/tasks/background_tasks/__init__.py",
        "main/agent/tasks/background_tasks/batch_analysis.py",
        "main/agent/tasks/background_tasks/cleanup_operations.py",
        "main/agent/tasks/background_tasks/data_processing.py",
        "main/agent/tasks/background_tasks/email_processing.py",
        "main/agent/tasks/background_tasks/file_handling.py",
        "main/agent/tasks/background_tasks/notification_sending.py",
        "main/agent/tasks/background_tasks/report_delivery.py",
        "main/agent/tasks/background_tasks/web_scraping.py",
        
        # main/agent/tasks/chem_tasks/
        "main/agent/tasks/chem_tasks/__init__.py",
        "main/agent/tasks/chem_tasks/batch_processing.py",
        "main/agent/tasks/chem_tasks/chemical_analysis.py",
        "main/agent/tasks/chem_tasks/data_validation.py",
        "main/agent/tasks/chem_tasks/molecule_processing.py",
        "main/agent/tasks/chem_tasks/property_prediction.py",
        "main/agent/tasks/chem_tasks/similarity_calculation.py",
        "main/agent/tasks/chem_tasks/toxicity_assessment.py",
        
        # main/agent/tasks/kag_tasks/
        "main/agent/tasks/kag_tasks/__init__.py",
        "main/agent/tasks/kag_tasks/entity_extraction.py",
        "main/agent/tasks/kag_tasks/graph_building.py",
        "main/agent/tasks/kag_tasks/graph_embedding.py",
        "main/agent/tasks/kag_tasks/graph_maintenance.py",
        "main/agent/tasks/kag_tasks/inference_processing.py",
        "main/agent/tasks/kag_tasks/kg_query_processing.py",
        "main/agent/tasks/kag_tasks/relationship_mining.py",
        "main/agent/tasks/kag_tasks/visualization_generation.py",
        
        # main/agent/tasks/llm_tasks/
        "main/agent/tasks/llm_tasks/__init__.py",
        "main/agent/tasks/llm_tasks/cache_management.py",
        "main/agent/tasks/llm_tasks/cost_calculation.py",
        "main/agent/tasks/llm_tasks/fine_tuning.py",
        "main/agent/tasks/llm_tasks/llm_inference.py",
        "main/agent/tasks/llm_tasks/model_evaluation.py",
        "main/agent/tasks/llm_tasks/performance_testing.py",
        "main/agent/tasks/llm_tasks/prompt_engineering.py",
        "main/agent/tasks/llm_tasks/response_processing.py",
        
        # main/agent/tasks/periodic_tasks/
        "main/agent/tasks/periodic_tasks/__init__.py",
        "main/agent/tasks/periodic_tasks/backup_creation.py",
        "main/agent/tasks/periodic_tasks/cache_cleanup.py",
        "main/agent/tasks/periodic_tasks/data_sync.py",
        "main/agent/tasks/periodic_tasks/graph_refresh.py",
        "main/agent/tasks/periodic_tasks/health_check.py",
        "main/agent/tasks/periodic_tasks/index_update.py",
        "main/agent/tasks/periodic_tasks/maintenance_tasks.py",
        "main/agent/tasks/periodic_tasks/model_retraining.py",
        "main/agent/tasks/periodic_tasks/performance_reporting.py",
        
        # main/agent/tasks/rag_tasks/
        "main/agent/tasks/rag_tasks/__init__.py",
        "main/agent/tasks/rag_tasks/cache_updating.py",
        "main/agent/tasks/rag_tasks/embedding_generation.py",
        "main/agent/tasks/rag_tasks/index_maintenance.py",
        "main/agent/tasks/rag_tasks/indexing.py",
        "main/agent/tasks/rag_tasks/knowledge_base_update.py",
        "main/agent/tasks/rag_tasks/vector_search.py",
        
        # main/agent/tasks/voice_tasks/
        "main/agent/tasks/voice_tasks/__init__.py",
        "main/agent/tasks/voice_tasks/audio_processing.py",
        "main/agent/tasks/voice_tasks/speech_recognition.py",
        "main/agent/tasks/voice_tasks/speech_synthesis.py",
        "main/agent/tasks/voice_tasks/voice_command_processing.py",
        "main/agent/tasks/voice_tasks/wake_word_training.py",
        
        # main/agent/tts/
        "main/agent/tts/__init__.py",
        "main/agent/tts/openai_tts_engine.py",
        "main/agent/tts/google_tts_engine.py",
        "main/agent/tts/azure_tts_engine.py",
        "main/agent/tts/elevenlabs_engine.py",
        "main/agent/tts/tts_engine_manager.py",
        "main/agent/tts/prosody_controller.py",
        
        # main/agent/utils/
        "main/agent/utils/__init__.py",
        "main/agent/utils/data_utils.py",
        "main/agent/utils/error_utils.py",
        "main/agent/utils/file_utils.py",
        "main/agent/utils/logging_utils.py",
        "main/agent/utils/security_utils.py",
        "main/agent/utils/text_utils.py",
        "main/agent/utils/time_utils.py",
        "main/agent/utils/validation_utils.py",
        
        # main/agent/voice_ui/
        "main/agent/voice_ui/__init__.py",
        "main/agent/voice_ui/voice_command_processor.py",
        "main/agent/voice_ui/dialog_state_manager.py",
        "main/agent/voice_ui/voice_feedback_generator.py",
        "main/agent/voice_ui/audio_ui_integrator.py",
        
        # main/agent/web/
        "main/agent/web/__init__.py",
        "main/agent/web/cache_manager.py",
        "main/agent/web/content_parser.py",
        "main/agent/web/link_crawler.py",
        "main/agent/web/rate_limiter.py",
        "main/agent/web/robots_checker.py",
        "main/agent/web/scraper_engine.py",
        "main/agent/web/web_client.py",
        "main/agent/web/web_content_validator.py",
        
        # main/api/
        "main/api/__init__.py",
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
        
        # main/api/websocket/
        "main/api/websocket/__init__.py",
        "main/api/websocket/consumers.py",
        "main/api/websocket/middleware.py",
        "main/api/websocket/routing.py",
        "main/api/websocket/voice_consumers.py",
        
        # main/api/v1/
        "main/api/v1/__init__.py",
        "main/api/v1/serializers.py",
        "main/api/v1/tests.py",
        "main/api/v1/urls.py",
        "main/api/v1/views.py",
        
        # main/api/v1/agents/
        "main/api/v1/agents/__init__.py",
        "main/api/v1/agents/serializers.py",
        "main/api/v1/agents/urls.py",
        "main/api/v1/agents/views.py",
        
        # main/api/v1/chat/
        "main/api/v1/chat/__init__.py",
        "main/api/v1/chat/serializers.py",
        "main/api/v1/chat/urls.py",
        "main/api/v1/chat/views.py",
        
        # main/api/v1/chemical/
        "main/api/v1/chemical/__init__.py",
        "main/api/v1/chemical/serializers.py",
        "main/api/v1/chemical/urls.py",
        "main/api/v1/chemical/views.py",
        
        # main/api/v1/data/
        "main/api/v1/data/__init__.py",
        "main/api/v1/data/serializers.py",
        "main/api/v1/data/urls.py",
        "main/api/v1/data/views.py",
        
        # main/api/v1/kag/
        "main/api/v1/kag/__init__.py",
        "main/api/v1/kag/serializers.py",
        "main/api/v1/kag/urls.py",
        "main/api/v1/kag/views.py",
        
        # main/api/v1/multilingual/
        "main/api/v1/multilingual/__init__.py",
        "main/api/v1/multilingual/serializers.py",
        "main/api/v1/multilingual/urls.py",
        "main/api/v1/multilingual/views.py",
        
        # main/api/v1/rag/
        "main/api/v1/rag/__init__.py",
        "main/api/v1/rag/serializers.py",
        "main/api/v1/rag/urls.py",
        "main/api/v1/rag/views.py",
        
        # main/api/v1/voice/
        "main/api/v1/voice/__init__.py",
        "main/api/v1/voice/serializers.py",
        "main/api/v1/voice/urls.py",
        "main/api/v1/voice/views.py",
        
        # main/migrations/
        "main/migrations/__init__.py",
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
        "main/migrations/0010_voice_models.py",
        
        # main/static/main/css/
        "main/static/main/css/agent-chat.css",
        "main/static/main/css/chat-interface.css",
        "main/static/main/css/citation-styles.css",
        "main/static/main/css/loading-animations.css",
        "main/static/main/css/message-bubble.css",
        "main/static/main/css/responsive-chat.css",
        "main/static/main/css/rtl-support.css",
        "main/static/main/css/voice-interface.css",
        
        # main/static/main/js/
        "main/static/main/js/agent-chat.js",
        "main/static/main/js/agent-control.js",
        "main/static/main/js/audio-effects.js",
        "main/static/main/js/chat-streaming.js",
        "main/static/main/js/citation-display.js",
        "main/static/main/js/file-upload.js",
        "main/static/main/js/language-switcher.js",
        "main/static/main/js/markdown-render.js",
        "main/static/main/js/message-handler.js",
        "main/static/main/js/realtime-updates.js",
        "main/static/main/js/typing-simulator.js",
        "main/static/main/js/voice-commands.js",
        "main/static/main/js/voice-player.js",
        "main/static/main/js/voice-recorder.js",
        
        # main/templates/main/
        "main/templates/main/agent_chat.html",
        "main/templates/main/agent_settings.html",
        "main/templates/main/audio_message.html",
        "main/templates/main/base_agent.html",
        "main/templates/main/chat_messages.html",
        "main/templates/main/chat_sidebar.html",
        "main/templates/main/citation_display.html",
        "main/templates/main/conversation_history.html",
        "main/templates/main/file_upload.html",
        "main/templates/main/language_selector.html",
        "main/templates/main/loading_indicator.html",
        "main/templates/main/message_bubble.html",
        "main/templates/main/search_results.html",
        "main/templates/main/source_references.html",
        "main/templates/main/voice_interface.html",
        "main/templates/main/voice_settings.html",
        
        # requirements/
        "requirements/requirements-chem.txt",
        "requirements/requirements-dev.txt",
        "requirements/requirements-kag.txt",
        "requirements/requirements-llm.txt",
        "requirements/requirements-prod.txt",
        "requirements/requirements-rag.txt",
        "requirements/requirements-test.txt",
        "requirements/requirements-vision.txt",
        "requirements/requirements-voice.txt",
        "requirements/requirements.txt",
        
        # tests/
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/pytest.ini",
        
        # tests/benchmarks/
        "tests/benchmarks/__init__.py",
        "tests/benchmarks/benchmark_agents.py",
        "tests/benchmarks/benchmark_chem.py",
        "tests/benchmarks/benchmark_integration.py",
        "tests/benchmarks/benchmark_kag.py",
        "tests/benchmarks/benchmark_llm.py",
        "tests/benchmarks/benchmark_rag.py",
        "tests/benchmarks/benchmark_voice.py",
        
        # tests/test_agent/
        "tests/test_agent/__init__.py",
        "tests/test_agent/test_agent_core.py",
        "tests/test_agent/test_agent_integration.py",
        "tests/test_agent/test_chat_agent.py",
        "tests/test_agent/test_chronobiotics_agent.py",
        "tests/test_agent/test_citation_system.py",
        "tests/test_agent/test_parallel_execution.py",
        "tests/test_agent/test_response_formatter.py",
        "tests/test_agent/test_search_agents.py",
        
        # tests/test_chem/
        "tests/test_chem/__init__.py",
        "tests/test_chem/test_chemical_integration.py",
        "tests/test_chem/test_chemical_parser.py",
        "tests/test_chem/test_chemical_service.py",
        "tests/test_chem/test_img2mol.py",
        "tests/test_chem/test_molecule_analyzer.py",
        "tests/test_chem/test_properties_calculator.py",
        "tests/test_chem/test_similarity_calculator.py",
        
        # tests/test_kag/
        "tests/test_kag/__init__.py",
        "tests/test_kag/test_entity_extraction.py",
        "tests/test_kag/test_graph_builder.py",
        "tests/test_kag/test_graph_query.py",
        "tests/test_kag/test_inference_engine.py",
        "tests/test_kag/test_kag_integration.py",
        "tests/test_kag/test_kg_retriever.py",
        "tests/test_kag/test_knowledge_graph.py",
        
        # tests/test_llm/
        "tests/test_llm/__init__.py",
        "tests/test_llm/test_fine_tuning.py",
        "tests/test_llm/test_llm_integration.py",
        "tests/test_llm/test_llm_models.py",
        "tests/test_llm/test_multimodal_agents.py",
        "tests/test_llm/test_multimodal_llm.py",
        "tests/test_llm/test_prompt_engineering.py",
        "tests/test_llm/test_tools.py",
        
        # tests/test_rag/
        "tests/test_rag/__init__.py",
        "tests/test_rag/test_chunking.py",
        "tests/test_rag/test_embedding.py",
        "tests/test_rag/test_index_builder.py",
        "tests/test_rag/test_rag_integration.py",
        "tests/test_rag/test_reranker.py",
        "tests/test_rag/test_retriever.py",
        "tests/test_rag/test_vector_store.py",
        
        # tests/test_tasks/
        "tests/test_tasks/__init__.py",
        "tests/test_tasks/test_agent_tasks.py",
        "tests/test_tasks/test_chem_tasks.py",
        "tests/test_tasks/test_kag_tasks.py",
        "tests/test_tasks/test_llm_tasks.py",
        "tests/test_tasks/test_periodic_tasks.py",
        "tests/test_tasks/test_rag_tasks.py",
        
        # tests/test_utils/
        "tests/test_utils/__init__.py",
        "tests/test_utils/test_data_processing.py",
        "tests/test_utils/test_error_handling.py",
        "tests/test_utils/test_file_handling.py",
        "tests/test_utils/test_logging.py",
        "tests/test_utils/test_monitoring.py",
        "tests/test_utils/test_security.py",
        "tests/test_utils/test_time_utils.py",
        
        # utils/
        "utils/__init__.py",
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
        
        # utils/data_processing/
        "utils/data_processing/__init__.py",
        "utils/data_processing/data_cleaner.py",
        "utils/data_processing/data_filter.py",
        "utils/data_processing/data_formatter.py",
        "utils/data_processing/data_normalizer.py",
        "utils/data_processing/data_quality.py",
        "utils/data_processing/data_serializer.py",
        "utils/data_processing/data_transformer.py",
        "utils/data_processing/data_validator.py",
        
        # utils/error_handling/
        "utils/error_handling/__init__.py",
        "utils/error_handling/circuit_breaker.py",
        "utils/error_handling/error_codes.py",
        "utils/error_handling/error_recovery.py",
        "utils/error_handling/error_reporter.py",
        "utils/error_handling/exception_handler.py",
        "utils/error_handling/fallback_handler.py",
        "utils/error_handling/graceful_degradation.py",
        "utils/error_handling/retry_manager.py",
        
        # utils/file_handling/
        "utils/file_handling/__init__.py",
        "utils/file_handling/archive_handler.py",
        "utils/file_handling/backup_manager.py",
        "utils/file_handling/downloader.py",
        "utils/file_handling/file_converter.py",
        "utils/file_handling/file_manager.py",
        "utils/file_handling/file_storage.py",
        "utils/file_handling/file_validator.py",
        "utils/file_handling/uploader.py",
        
        # utils/logging/
        "utils/logging/__init__.py",
        "utils/logging/audit_logger.py",
        "utils/logging/error_logger.py",
        "utils/logging/log_analyzer.py",
        "utils/logging/log_formatter.py",
        "utils/logging/log_handler.py",
        "utils/logging/log_rotation.py",
        "utils/logging/metrics_logger.py",
        "utils/logging/performance_logger.py",
        
        # utils/monitoring/
        "utils/monitoring/__init__.py",
        "utils/monitoring/alert_manager.py",
        "utils/monitoring/dashboard_generator.py",
        "utils/monitoring/health_check.py",
        "utils/monitoring/metrics.py",
        "utils/monitoring/performance_monitor.py",
        "utils/monitoring/resource_tracker.py",
        "utils/monitoring/system_monitor.py",
        "utils/monitoring/tracing.py",
        "utils/monitoring/usage_tracker.py",
        
        # utils/network/
        "utils/network/__init__.py",
        "utils/network/api_client.py",
        "utils/network/connection_pool.py",
        "utils/network/dns_resolver.py",
        "utils/network/http_client.py",
        "utils/network/network_monitor.py",
        "utils/network/proxy_manager.py",
        "utils/network/websocket_client.py",
        
        # utils/security/
        "utils/security/__init__.py",
        "utils/security/access_control.py",
        "utils/security/authentication.py",
        "utils/security/authorization.py",
        "utils/security/encryption.py",
        "utils/security/input_validator.py",
        "utils/security/rate_limiter.py",
        "utils/security/sanitizer.py",
        "utils/security/security_audit.py",
        "utils/security/token_manager.py",
        
        # utils/time/
        "utils/time/__init__.py",
        "utils/time/cache_expiry.py",
        "utils/time/cron_parser.py",
        "utils/time/date_parser.py",
        "utils/time/rate_limiter.py",
        "utils/time/scheduler.py",
        "utils/time/time_utils.py",
        "utils/time/timezone_handler.py",
    ]
    
    print(f"Создание структуры проекта в: {os.path.abspath(base_path)}")
    
    # Создание директорий
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"  Создана директория: {directory}")
    
    # Создание файлов
    for file in files:
        file_path = os.path.join(base_path, file)
        
        # Определяем начальное содержимое файла на основе его типа
        content = ""
        if file.endswith('.py'):
            if '__init__.py' in file:
                content = '''"""Init file."""\n\n__version__ = "1.0.0"\n'''
            elif file in ['manage.py', 'chronobiotic/wsgi.py', 'chronobiotic/asgi.py']:
                # Django файлы с минимальным содержимым
                content = '''import os\nimport sys\n\nfrom django.core.wsgi import get_wsgi_application\n\nos.environ.setdefault("DJANGO_SETTINGS_MODULE", "chronobiotic.settings")\n\napplication = get_wsgi_application()\n'''
            else:
                content = '''"""Module for ..."""\n\n# Placeholder file\n\n'''
        elif file.endswith('.json'):
            content = '[]'
        elif file.endswith('.html'):
            content = '''<!-- Template placeholder -->\n\n{% extends "base.html" %}\n\n{% block content %}\n    <div>Content goes here</div>\n{% endblock %}\n'''
        elif file.endswith('.css'):
            content = '/* CSS placeholder */\n'
        elif file.endswith('.js'):
            content = '// JavaScript placeholder\n'
        elif file == 'README.md':
            content = '''# Chronobiotic Agent\n\nA comprehensive Django-based agent system for chronobiotics research.\n'''
        elif file == '.gitignore':
            content = '''# Django
*.log
*.pot
*.pyc
__pycache__/
local_settings.py
db.sqlite3
db.sqlite3-journal
media/
staticfiles/

# Virtual Environment
venv/
env/
ENV/
.env
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# OS
Thumbs.db

# Coverage
.coverage
htmlcov/

# Distribution / packaging
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
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
'''
        elif file == 'requirements.txt':
            content = '''Django>=4.2
djangorestframework>=3.14
celery>=5.3
redis>=4.6
psycopg2-binary>=2.9
django-cors-headers>=4.2
'''
        elif file == 'docker-compose.yml':
            content = '''version: '3.8'

services:
  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - redis
      - db

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: chronobiotic
      POSTGRES_USER: chronobiotic_user
      POSTGRES_PASSWORD: changeme

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  celery:
    build: .
    command: celery -A chronobioticagent worker -l info
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      - redis
      - db

  celery-beat:
    build: .
    command: celery -A chronobioticagent beat -l info
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      - redis
      - db

volumes:
  postgres_data:
'''
        elif file == 'Dockerfile':
            content = '''FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
'''
        elif file == '.env.example':
            content = '''# Django
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

# Database
DB_NAME=chronobiotic
DB_USER=chronobiotic_user
DB_PASSWORD=changeme
DB_HOST=db
DB_PORT=5432

# Redis
REDIS_URL=redis://redis:6379/0

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# LLM APIs (пример)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# External APIs
PUBCHEM_API_URL=https://pubchem.ncbi.nlm.nih.gov/rest/pug
CHEMBL_API_URL=https://www.ebi.ac.uk/chembl/api/data

# Email
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
'''
        
        # Создаем файл с содержимым
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Создан файл: {file}")
    
    print(f"\nСтруктура проекта успешно создана в: {os.path.abspath(base_path)}")
    
    # Создаем файл проверки структуры
    create_structure_checker(base_path)


def create_structure_checker(base_path):
    """Создает скрипт для проверки структуры проекта"""
    
    checker_content = '''#!/usr/bin/env python3
"""
Скрипт для проверки структуры проекта ChronobioticAgent.
Проверяет наличие всех необходимых директорий и файлов.
"""

import os
import sys
from pathlib import Path

def check_project_structure(base_path="."):
    """Проверяет структуру проекта"""

    required_dirs = [
        "chronobiotic",
        "fixtures",
        "main",
        "main/agent",
        "main/agent/agents",
        "main/agent/agents/analysis_agents",
        "main/agent/agents/assistant_agents",
        "main/agent/agents/chronobiotics_agents",
        "main/agent/agents/citation_agents",
        "main/agent/agents/data_agents",
        "main/agent/agents/multilingual_agents",
        "main/agent/agents/research_agents",
        "main/agent/agents/voice_agents",
        "main/agent/analysis",
        "main/agent/audio",
        "main/agent/chat",
        "main/agent/chem",
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
        "main/agent/geo",
        "main/agent/kag",
        "main/agent/kag/algorithms",
        "main/agent/kag/models",
        "main/agent/kag/queries",
        "main/agent/kag/storage",
        "main/agent/kag/utils",
        "main/agent/language_models",
        "main/agent/llm",
        "main/agent/llm/fine_tuning",
        "main/agent/llm/models",
        "main/agent/llm/multimodal_agents",
        "main/agent/llm/multimodal_agents/embeddings",
        "main/agent/llm/multimodal_agents/fusion",
        "main/agent/llm/multimodal_agents/vision",
        "main/agent/llm/multimodal_llm",
        "main/agent/llm/optimization",
        "main/agent/llm/prompts",
        "main/agent/llm/tools",
        "main/agent/localization",
        "main/agent/management/commands",
        "main/agent/management/commands/agent_commands",
        "main/agent/management/commands/data_commands",
        "main/agent/management/commands/kag_commands",
        "main/agent/management/commands/llm_commands",
        "main/agent/management/commands/multilingual_commands",
        "main/agent/management/commands/rag_commands",
        "main/agent/management/commands/system_commands",
        "main/agent/management/commands/voice_commands",
        "main/agent/memory",
        "main/agent/parallel",
        "main/agent/rag",
        "main/agent/rag/chunking",
        "main/agent/rag/embeddings",
        "main/agent/rag/knowledge_base",
        "main/agent/rag/reranking",
        "main/agent/rag/retrievers",
        "main/agent/rag/vector_store",
        "main/agent/response",
        "main/agent/search",
        "main/agent/stt",
        "main/agent/tasks",
        "main/agent/tasks/agent_tasks",
        "main/agent/tasks/background_tasks",
        "main/agent/tasks/chem_tasks",
        "main/agent/tasks/kag_tasks",
        "main/agent/tasks/llm_tasks",
        "main/agent/tasks/periodic_tasks",
        "main/agent/tasks/rag_tasks",
        "main/agent/tasks/voice_tasks",
        "main/agent/tts",
        "main/agent/utils",
        "main/agent/voice_ui",
        "main/agent/web",
        "main/api",
        "main/api/v1",
        "main/api/v1/agents",
        "main/api/v1/chat",
        "main/api/v1/chemical",
        "main/api/v1/data",
        "main/api/v1/kag",
        "main/api/v1/multilingual",
        "main/api/v1/rag",
        "main/api/v1/voice",
        "main/api/websocket",
        "main/migrations",
        "main/static/main/css",
        "main/static/main/js",
        "main/templates/main",
        "media/chemical_structures",
        "media/documents",
        "media/user_uploads",
        "requirements",
        "tests",
        "tests/benchmarks",
        "tests/fixtures",
        "tests/test_agent",
        "tests/test_chem",
        "tests/test_kag",
        "tests/test_llm",
        "tests/test_rag",
        "tests/test_tasks",
        "tests/test_utils",
        "utils",
        "utils/data_processing",
        "utils/error_handling",
        "utils/file_handling",
        "utils/logging",
        "utils/monitoring",
        "utils/network",
        "utils/security",
        "utils/time",
    ]

    required_files = [
        "manage.py",
        "README.md",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        ".gitignore",

        "chronobiotic/__init__.py",
        "chronobiotic/settings.py",
        "chronobiotic/urls.py",
        "chronobiotic/wsgi.py",
        "chronobiotic/asgi.py",

        "main/__init__.py",
        "main/models.py",
        "main/views.py",
        "main/urls.py",
        "main/admin.py",
        "main/apps.py",

        "main/agent/__init__.py",
        "main/agent/agent_core.py",
        "main/agent/chronobiotics_agent.py",

        "main/api/__init__.py",
        "main/api/urls.py",
        "main/api/views.py",

        "tests/__init__.py",
        "tests/conftest.py",

        "utils/__init__.py",
    ]

    print("Проверка структуры проекта ChronobioticAgent...")
    print("=" * 60)

    all_good = True
    missing_dirs = []
    missing_files = []

    # Проверяем директории
    print("\\nПроверка директорий:")
    for directory in required_dirs:
        dir_path = Path(base_path) / directory
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} - отсутствует")
            missing_dirs.append(directory)
            all_good = False

    # Проверяем файлы
    print("\\nПроверка файлов:")
    for file in required_files:
        file_path = Path(base_path) / file
        if file_path.exists() and file_path.is_file():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - отсутствует")
            missing_files.append(file)
            all_good = False

    print("\\n" + "=" * 60)

    if all_good:
        print("✅ Структура проекта в полном порядке!")
        return 0
    else:
        print("⚠️  Обнаружены проблемы в структуре проекта:")
        if missing_dirs:
            print(f"\\nОтсутствующие директории ({len(missing_dirs)}):")
            for dir_name in missing_dirs:
                print(f"  - {dir_name}")

        if missing_files:
            print(f"\\nОтсутствующие файлы ({len(missing_files)}):")
            for file_name in missing_files:
                print(f"  - {file_name}")

        print("\\nДля создания недостающих элементов используйте:")
        print("  python create_structure.py")

        return 1

if __name__ == "__main__":
    # Проверяем текущую директорию или переданную в аргументе
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."

    exit_code = check_project_structure(base_path)
    sys.exit(exit_code)
'''
    
    checker_path = os.path.join(base_path, "check_structure.py")
    with open(checker_path, 'w', encoding='utf-8') as f:
        f.write(checker_content)
    
    # Делаем скрипт исполняемым (Unix)
    if os.name != 'nt':  # не Windows
        os.chmod(checker_path, 0o755)
    
    print(f"  Создан скрипт проверки структуры: check_structure.py")


if __name__ == "__main__":
    # Если скрипт запущен напрямую
    if len(sys.argv) > 1:
        create_project_structure(sys.argv[1])
    else:
        create_project_structure()
