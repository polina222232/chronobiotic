#!/usr/bin/env python3
"""
Скрипт для проверки структуры проекта ChronobioticAgent.
Проверяет наличие всех необходимых директорий и файлов.
"""

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
    print("\nПроверка директорий:")
    for directory in required_dirs:
        dir_path = Path(base_path) / directory
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} - отсутствует")
            missing_dirs.append(directory)
            all_good = False
    
    # Проверяем файлы
    print("\nПроверка файлов:")
    for file in required_files:
        file_path = Path(base_path) / file
        if file_path.exists() and file_path.is_file():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - отсутствует")
            missing_files.append(file)
            all_good = False
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("✅ Структура проекта в полном порядке!")
        return 0
    else:
        print("⚠️  Обнаружены проблемы в структуре проекта:")
        if missing_dirs:
            print(f"\nОтсутствующие директории ({len(missing_dirs)}):")
            for dir_name in missing_dirs:
                print(f"  - {dir_name}")
        
        if missing_files:
            print(f"\nОтсутствующие файлы ({len(missing_files)}):")
            for file_name in missing_files:
                print(f"  - {file_name}")
        
        print("\nДля создания недостающих элементов используйте:")
        print("  python create_structure.py")
        
        return 1


if __name__ == "__main__":
    # Проверяем текущую директорию или переданную в аргументе
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    exit_code = check_project_structure(base_path)
    sys.exit(exit_code)
