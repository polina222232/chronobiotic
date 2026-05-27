Chronobiotic Agent

A comprehensive Django-based agent system for chronobiotics research.
📋 Описание

Chronobiotic Agent — это мощная система на базе Django для исследования хронобиотиков. Система включает в себя:

    Django REST API — современный RESTful интерфейс
    Celery + Redis — асинхронные задачи и фоновая обработка
    LLM интеграция — поддержка OpenAI, Anthropic, Google Generative AI, Cohere
    RAG & Vector Search — ChromaDB, FAISS, Qdrant для семантического поиска
    Химический анализ — интеграция с PubChem и ChEMBL API
    PostgreSQL — надёжное хранилище данных

🚀 Быстрый старт
Требования

    Python 3.10+
    Docker и Docker Compose
    Git

Установка

    Клонируйте репозиторий:

    git clone <repository-url>
    cd chronobiotic-agent

    Настройте переменные окружения:

    cp .env.example .env
    # Отредактируйте .env и укажите ваши API ключи и настройки

    Запустите через Docker Compose:

    docker-compose up --build

    Откройте приложение:
        Web API: http://localhost:8000
        Django Admin: http://localhost:8000/admin

📁 Структура проекта

chronobiotic-agent/
├── main/               # Основное Django приложение
│   ├── agent/          # Модуль агентов
│   ├── api/            # REST API эндпоинты
│   ├── templates/      # HTML шаблоны
│   └── static/         # Статические файлы
├── chronobiotic/       # Настройки проекта Django
├── tests/              # Тесты
├── utils/              # Утилиты и вспомогательные модули
├── fixtures/           # Тестовые данные
├── requirements/       # Файлы зависимостей
└── manage.py           # Django management script

⚙️ Конфигурация
Переменные окружения

Основные переменные в .env:
Переменная 	Описание 	Пример
DJANGO_SECRET_KEY 	Секретный ключ Django 	your-secret-key
DJANGO_DEBUG 	Режим отладки 	True
DB_NAME 	Имя базы данных 	chronobiotic
DB_USER 	Пользователь БД 	solar
DB_PASSWORD 	Пароль БД 	solarnotfound
OPENAI_API_KEY 	API ключ OpenAI 	sk-...
ANTHROPIC_API_KEY 	API ключ Anthropic 	sk-ant-...
REDIS_URL 	URL Redis 	redis://redis:6379/0
🔧 Разработка
Локальная установка (без Docker)

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Миграции базы данных
python manage.py migrate

# Запуск сервера разработки
python manage.py runserver

Запуск Celery worker

celery -A chronobioticagent worker -l info

Запуск Celery beat (планировщик)

celery -A chronobioticagent beat -l info

Тестирование

# Запуск всех тестов
python manage.py test

# Запуск тестов с покрытием
pytest --cov=.

📚 API Документация

После запуска приложения документация Swagger доступна по адресу:

    http://localhost:8000/swagger/
    http://localhost:8000/redoc/

🧪 Основные возможности

    Агентная система — умные агенты для анализа данных
    Интеграция с LLM — поддержка множественных языковых моделей
    Векторный поиск — семантический поиск по научным данным
    Химические API — работа с PubChem и ChEMBL
    Асинхронные задачи — фоновая обработка через Celery
    REST API — полный набор эндпоинтов для интеграции

🤝 Вклад в проект

    Fork репозитория
    Создайте ветку (git checkout -b feature/amazing-feature)
    Commit изменения (git commit -m 'Add amazing feature')
    Push в ветку (git push origin feature/amazing-feature)
    Откройте Pull Request

