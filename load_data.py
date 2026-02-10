import os
import sys
import json
import django
from pathlib import Path
import shutil

# Добавляем путь к проекту Django
project_path = Path(__file__).resolve().parent
sys.path.append(str(project_path))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chronobiotic.settings')
django.setup()

from main.models import (
    Chronobiotic, Synonyms, Articles, Targets, Effect,
    Mechanism, Bioclass
)
from django.db import transaction, models, connection
from django.core.files import File


def reset_sequences():
    """Сброс последовательностей ID для всех таблиц"""
    print("Сброс последовательностей ID...")
    
    # Получаем все таблицы
    tables = [
        'synonyms', 'chronobiotic', 'article', 'target',
        'effect', 'mechanism', 'class'
    ]
    
    with connection.cursor() as cursor:
        for table in tables:
            try:
                cursor.execute(f"SELECT setval('{table}_id_seq', 1, false);")
                print(f"  Сброшена последовательность для {table}")
            except Exception as e:
                print(f"  Ошибка сброса последовательности для {table}: {e}")


def clear_database():
    """Полная очистка базы данных"""
    print("=== ОЧИСТКА БАЗЫ ДАННЫХ ===")
    
    # Удаляем в правильном порядке для избежания ошибок внешних ключей
    models_list = [Synonyms, Chronobiotic, Articles, Targets, Effect, Mechanism, Bioclass]
    
    for model in models_list:
        count = model.objects.count()
        model.objects.all().delete()
        print(f"Удалено {count} записей из {model._meta.db_table}")
    
    # Сбрасываем последовательности ID
    reset_sequences()


def load_json_data(json_file_path):
    """Загрузка данных из JSON файла"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки JSON: {e}")
        return None


def safe_get_field(fields, field_name, default=''):
    """Безопасное получение поля из JSON с обработкой None значений"""
    value = fields.get(field_name)
    if value is None:
        return default
    return str(value) if value is not None else default


def setup_media_directory():
    """Создание структуры медиа-папки"""
    media_dir = Path('media')
    media_dir.mkdir(exist_ok=True)
    
    # Создаем подпапки если нужно
    subdirs = ['molecules', 'temp']
    for subdir in subdirs:
        (media_dir / subdir).mkdir(exist_ok=True)
    
    print(f"Медиа-папка готова: {media_dir.absolute()}")


def find_image_file(image_path_from_json):
    """Поиск файла изображения по пути из JSON"""
    if not image_path_from_json or not isinstance(image_path_from_json, str):
        return None
    
    # Пробуем разные возможные расположения файлов
    possible_paths = [
        Path(image_path_from_json),  # Прямой путь из JSON
        Path('media') / image_path_from_json,  # В папке media
        Path('media') / Path(image_path_from_json).name,  # Только имя файла в media
        Path(image_path_from_json).name,  # Только имя файла в текущей директории
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            print(f"  Найдено изображение: {path}")
            return path
    
    print(f"  ⚠️ Изображение не найдено: {image_path_from_json}")
    return None


def copy_image_to_media(source_path, chronobiotic_name):
    """Копирование изображения в медиа-папку с правильным именем"""
    if not source_path or not source_path.exists():
        return None
    
    # Создаем безопасное имя файла
    safe_name = f"{chronobiotic_name}_{source_path.name}"
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    
    # Путь назначения
    dest_path = Path('media') / safe_name
    
    try:
        # Копируем файл
        shutil.copy2(source_path, dest_path)
        print(f"  ✅ Изображение скопировано: {dest_path}")
        return str(dest_path)
    except Exception as e:
        print(f"  ❌ Ошибка копирования изображения: {e}")
        return None


def create_objects_with_original_ids(json_data):
    """Создание объектов с сохранением оригинальных ID и изображений"""
    
    print("\n=== ЗАГРУЗКА ДАННЫХ С СОХРАНЕНИЕМ ID И ИЗОБРАЖЕНИЙ ===")
    
    # Подготавливаем медиа-папку
    setup_media_directory()
    
    # Словари для хранения созданных объектов
    objects_dict = {
        'main.articles': {},
        'main.targets': {},
        'main.mechanism': {},
        'main.effect': {},
        'main.bioclass': {},
        'main.chronobiotic': {},
        'main.synonyms': {},
    }
    
    # Статистика
    stats = {
        'main.articles': {'created': 0, 'errors': 0},
        'main.targets': {'created': 0, 'errors': 0},
        'main.mechanism': {'created': 0, 'errors': 0},
        'main.effect': {'created': 0, 'errors': 0},
        'main.bioclass': {'created': 0, 'errors': 0},
        'main.chronobiotic': {'created': 0, 'errors': 0, 'images_loaded': 0},
        'main.synonyms': {'created': 0, 'errors': 0},
    }
    
    # Фаза 1: Создание основных объектов с оригинальными ID
    print("Фаза 1: Создание основных объектов с оригинальными ID...")
    
    # Сначала создаем все простые объекты
    simple_models = [
        ('main.articles', Articles),
        ('main.targets', Targets),
        ('main.mechanism', Mechanism),
        ('main.effect', Effect),
        ('main.bioclass', Bioclass),
    ]
    
    for model_name, model_class in simple_models:
        print(f"Загрузка {model_name}...")
        
        for item in json_data:
            if item.get('model') != model_name:
                continue
            
            pk = item.get('pk')
            fields = item.get('fields', {})
            
            try:
                if model_name == 'main.articles':
                    articlename = safe_get_field(fields, 'articlename')
                    if not articlename or articlename.strip() == '':
                        articlename = f"Article_{pk}"
                    
                    # Создаем объект с указанием ID
                    obj = model_class(
                        id=pk,
                        articlename=articlename,
                        articleurl=safe_get_field(fields, 'articleurl')
                    )
                    obj.save()
                    objects_dict[model_name][pk] = obj
                    stats[model_name]['created'] += 1
                
                elif model_name == 'main.targets':
                    targetsname = safe_get_field(fields, 'targetsname')
                    if not targetsname or targetsname.strip() == '':
                        targetsname = f"Target_{pk}"
                    
                    obj = model_class(
                        id=pk,
                        targetsname=targetsname,
                        targetsfullname=safe_get_field(fields, 'targetsfullname'),
                        targeturl=safe_get_field(fields, 'targeturl'),
                    )
                    obj.save()
                    objects_dict[model_name][pk] = obj
                    stats[model_name]['created'] += 1
                
                elif model_name == 'main.mechanism':
                    mechanismname = safe_get_field(fields, 'mechanismname')
                    if not mechanismname or mechanismname.strip() == '':
                        mechanismname = f"Mechanism_{pk}"
                    
                    obj = model_class(
                        id=pk,
                        mechanismname=mechanismname
                    )
                    obj.save()
                    objects_dict[model_name][pk] = obj
                    stats[model_name]['created'] += 1
                
                elif model_name == 'main.effect':
                    effectname = safe_get_field(fields, 'Effectname')
                    if not effectname or effectname.strip() == '':
                        effectname = f"Effect_{pk}"
                    
                    obj = model_class(
                        id=pk,
                        Effectname=effectname
                    )
                    obj.save()
                    objects_dict[model_name][pk] = obj
                    stats[model_name]['created'] += 1
                
                elif model_name == 'main.bioclass':
                    nameclass = safe_get_field(fields, 'nameclass')
                    if not nameclass or nameclass.strip() == '':
                        nameclass = f"Class_{pk}"
                    
                    obj = model_class(
                        id=pk,
                        nameclass=nameclass
                    )
                    obj.save()
                    objects_dict[model_name][pk] = obj
                    stats[model_name]['created'] += 1
            
            except Exception as e:
                print(f"Ошибка при создании {model_name} (PK: {pk}): {e}")
                stats[model_name]['errors'] += 1
    
    # Затем создаем хронобиотики с изображениями
    print("Загрузка main.chronobiotic с изображениями...")
    for item in json_data:
        if item.get('model') != 'main.chronobiotic':
            continue
        
        pk = item.get('pk')
        fields = item.get('fields', {})
        
        try:
            gname = safe_get_field(fields, 'gname')
            if not gname or gname.strip() == '':
                stats['main.chronobiotic']['errors'] += 1
                continue
            
            # Создаем хронобиотик БЕЗ изображения сначала
            obj = Chronobiotic(
                id=pk,
                gname=gname,
                smiles=safe_get_field(fields, 'smiles', 'Not specified')[:256],
                linkname=safe_get_field(fields, 'linkname', '')[:256],
                molecula=safe_get_field(fields, 'molecula', 'Not specified')[:256],
                iupacname=safe_get_field(fields, 'iupacname', 'Not specified')[:256],
                description=safe_get_field(fields, 'description', '')[:5000],
                fdastatus=safe_get_field(fields, 'fdastatus', '')[:64],
                linkslists=safe_get_field(fields, 'linkslists', ''),
                pubchem=safe_get_field(fields, 'pubchem', ''),
                chemspider=safe_get_field(fields, 'chemspider', ''),
                drugbank=safe_get_field(fields, 'drugbank', ''),
                chebi=safe_get_field(fields, 'chebi', ''),
                uniprot=safe_get_field(fields, 'uniprot', ''),
                kegg=safe_get_field(fields, 'kegg', ''),
                selleckchem=safe_get_field(fields, 'selleckchem', ''),
            )
            
            # Сохраняем объект сначала без изображения
            obj.save()
            
            # Затем обрабатываем изображение отдельно
            image_path_from_json = safe_get_field(fields, 'updphoto')
            
            if image_path_from_json and image_path_from_json.strip():
                source_image_path = find_image_file(image_path_from_json)
                if source_image_path:
                    # Копируем изображение в медиа-папку
                    media_image_path = copy_image_to_media(source_image_path, gname)
                    if media_image_path:
                        # Открываем файл и прикрепляем к объекту
                        with open(media_image_path, 'rb') as f:
                            image_file = File(f, name=Path(media_image_path).name)
                            obj.updphoto.save(image_file.name, image_file, save=True)
                        stats['main.chronobiotic']['images_loaded'] += 1
                        print(f"  ✅ Изображение прикреплено к: {gname}")
            
            objects_dict['main.chronobiotic'][pk] = obj
            stats['main.chronobiotic']['created'] += 1
            
            if stats['main.chronobiotic']['created'] % 100 == 0:
                print(f"  Загружено {stats['main.chronobiotic']['created']} хронобиотиков...")
        
        except Exception as e:
            print(f"Ошибка при создании хронобиотика (PK: {pk}): {e}")
            stats['main.chronobiotic']['errors'] += 1
    
    # Фаза 2: Установка связей ManyToMany
    print("\nФаза 2: Установка связей...")
    
    connection_stats = {
        'articles_connections': 0,
        'targets_connections': 0,
        'mechanisms_connections': 0,
        'effects_connections': 0,
        'classes_connections': 0,
        'synonyms_connections': 0,
    }
    
    # Устанавливаем связи для хронобиотиков
    for item in json_data:
        if item.get('model') != 'main.chronobiotic':
            continue
        
        pk = item.get('pk')
        fields = item.get('fields', {})
        
        try:
            chronobiotic = objects_dict['main.chronobiotic'].get(pk)
            if not chronobiotic:
                continue
            
            # Статьи
            if 'articles' in fields and fields['articles']:
                article_ids = fields['articles']
                articles = []
                for aid in article_ids:
                    if aid in objects_dict['main.articles']:
                        articles.append(objects_dict['main.articles'][aid])
                if articles:
                    chronobiotic.articles.set(articles)
                    connection_stats['articles_connections'] += len(articles)
            
            # Мишени
            if 'target' in fields and fields['target']:
                target_ids = fields['target']
                targets = []
                for tid in target_ids:
                    if tid in objects_dict['main.targets']:
                        targets.append(objects_dict['main.targets'][tid])
                if targets:
                    chronobiotic.target.set(targets)
                    connection_stats['targets_connections'] += len(targets)
            
            # Механизмы
            if 'mechanisms' in fields and fields['mechanisms']:
                mechanism_ids = fields['mechanisms']
                mechanisms = []
                for mid in mechanism_ids:
                    if mid in objects_dict['main.mechanism']:
                        mechanisms.append(objects_dict['main.mechanism'][mid])
                if mechanisms:
                    chronobiotic.mechanisms.set(mechanisms)
                    connection_stats['mechanisms_connections'] += len(mechanisms)
            
            # Эффекты
            if 'effect' in fields and fields['effect']:
                effect_ids = fields['effect']
                effects = []
                for eid in effect_ids:
                    if eid in objects_dict['main.effect']:
                        effects.append(objects_dict['main.effect'][eid])
                if effects:
                    chronobiotic.effect.set(effects)
                    connection_stats['effects_connections'] += len(effects)
            
            # Классы
            if 'classf' in fields and fields['classf']:
                class_ids = fields['classf']
                bioclasses = []
                for cid in class_ids:
                    if cid in objects_dict['main.bioclass']:
                        bioclasses.append(objects_dict['main.bioclass'][cid])
                if bioclasses:
                    chronobiotic.classf.set(bioclasses)
                    connection_stats['classes_connections'] += len(bioclasses)
        
        except Exception as e:
            print(f"Ошибка при установке связей для хронобиотика (PK: {pk}): {e}")
    
    # Создаем синонимы
    print("Загрузка main.synonyms...")
    for item in json_data:
        if item.get('model') != 'main.synonyms':
            continue
        
        pk = item.get('pk')
        fields = item.get('fields', {})
        
        try:
            synonym_name = safe_get_field(fields, 'synonymsmname')
            chronobiotic_id = fields.get('originalbiotic')
            
            if not synonym_name or synonym_name.strip() == '':
                continue
            
            if synonym_name and chronobiotic_id:
                chronobiotic = objects_dict['main.chronobiotic'].get(chronobiotic_id)
                if chronobiotic:
                    obj = Synonyms(
                        id=pk,
                        synonymsmname=synonym_name,
                        originalbiotic=chronobiotic
                    )
                    obj.save()
                    objects_dict['main.synonyms'][pk] = obj
                    stats['main.synonyms']['created'] += 1
                    connection_stats['synonyms_connections'] += 1
        
        except Exception as e:
            print(f"Ошибка при создании синонима (PK: {pk}): {e}")
            stats['main.synonyms']['errors'] += 1
    
    return stats, connection_stats


def verify_loaded_data(json_data):
    """Проверка загруженных данных"""
    print("\n=== ПРОВЕРКА ДАННЫХ ===")
    
    models_stats = {
        'Хронобиотики': Chronobiotic.objects.count(),
        'Синонимы': Synonyms.objects.count(),
        'Статьи': Articles.objects.count(),
        'Мишени': Targets.objects.count(),
        'Механизмы': Mechanism.objects.count(),
        'Эффекты': Effect.objects.count(),
        'Биоклассы': Bioclass.objects.count(),
    }
    
    for model_name, count in models_stats.items():
        print(f"{model_name}: {count}")
    
    # Проверяем хронобиотики на наличие обязательных полей
    print("\nПроверка целостности хронобиотиков:")
    problematic = Chronobiotic.objects.filter(
        models.Q(smiles='') |
        models.Q(molecula='') |
        models.Q(iupacname='')
    )
    
    if problematic.exists():
        print(f"Найдено хронобиотиков с пустыми обязательными полями: {problematic.count()}")
        for cb in problematic[:5]:
            print(f"  - {cb.gname}: smiles='{cb.smiles}', molecula='{cb.molecula}', iupacname='{cb.iupacname}'")
    else:
        print("Все хронобиотики имеют обязательные поля")
    
    # Проверяем изображения
    print("\nПроверка изображений:")
    chronobiotics_with_images = Chronobiotic.objects.exclude(updphoto='').count()
    total_chronobiotics = Chronobiotic.objects.count()
    print(f"Хронобиотиков с изображениями: {chronobiotics_with_images}/{total_chronobiotics}")
    
    # Проверяем связи ManyToMany
    print("\nПроверка связей:")
    chronobiotics_with_articles = Chronobiotic.objects.filter(articles__isnull=False).distinct().count()
    chronobiotics_with_targets = Chronobiotic.objects.filter(target__isnull=False).distinct().count()
    chronobiotics_with_mechanisms = Chronobiotic.objects.filter(mechanisms__isnull=False).distinct().count()
    chronobiotics_with_effects = Chronobiotic.objects.filter(effect__isnull=False).distinct().count()
    chronobiotics_with_classes = Chronobiotic.objects.filter(classf__isnull=False).distinct().count()
    
    print(f"Хронобиотиков со статьями: {chronobiotics_with_articles}")
    print(f"Хронобиотиков с мишенями: {chronobiotics_with_targets}")
    print(f"Хронобиотиков с механизмами: {chronobiotics_with_mechanisms}")
    print(f"Хронобиотиков с эффектами: {chronobiotics_with_effects}")
    print(f"Хронобиотиков с классами: {chronobiotics_with_classes}")


def main():
    """Основная функция"""
    
    json_file_path = 'db0311.json'
    
    # Проверяем существование файла
    if not os.path.exists(json_file_path):
        print(f"Файл {json_file_path} не найден!")
        return
    
    # Загрузка JSON данных
    print("Загрузка JSON файла...")
    json_data = load_json_data(json_file_path)
    if not json_data:
        return
    
    print(f"Загружено {len(json_data)} записей из JSON")
    
    # Подсчет записей по моделям в JSON
    json_stats = {}
    for item in json_data:
        model_name = item.get('model')
        json_stats[model_name] = json_stats.get(model_name, 0) + 1
    
    print("Распределение в JSON файле:")
    for model_name, count in json_stats.items():
        print(f"  {model_name}: {count}")
    
    # Очистка базы данных
    clear_database()
    
    # Загрузка данных с транзакцией для безопасности
    try:
        with transaction.atomic():
            stats, connection_stats = create_objects_with_original_ids(json_data)
        
        print("\n=== ДЕТАЛЬНАЯ СТАТИСТИКА ЗАГРУЗКИ ===")
        print("Создано объектов:")
        for model_name, stat in stats.items():
            print(f"  {model_name}:")
            print(f"    Создано: {stat['created']}")
            if model_name == 'main.chronobiotic':
                print(f"    Изображений загружено: {stat['images_loaded']}")
            print(f"    Ошибок: {stat['errors']}")
        
        print(f"\nУстановлено связей:")
        for connection_type, count in connection_stats.items():
            print(f"  {connection_type}: {count}")
        
        # Проверка загруженных данных
        verify_loaded_data(json_data)
        
        # Финальная проверка
        print(f"\n=== ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ===")
        
        # Проверяем все модели
        all_models_loaded = True
        missing_details = []
        
        for model_name, expected_count in json_stats.items():
            if model_name.startswith('main.'):
                model_class = {
                    'main.articles': Articles,
                    'main.targets': Targets,
                    'main.mechanism': Mechanism,
                    'main.effect': Effect,
                    'main.bioclass': Bioclass,
                    'main.chronobiotic': Chronobiotic,
                    'main.synonyms': Synonyms,
                }.get(model_name)
                
                if model_class:
                    actual_count = model_class.objects.count()
                    status = "✅" if actual_count == expected_count else "❌"
                    print(f"  {model_name}: {actual_count}/{expected_count} {status}")
                    
                    if actual_count != expected_count:
                        all_models_loaded = False
                        missing_details.append(f"{model_name}: не хватает {expected_count - actual_count} записей")
        
        if all_models_loaded:
            print("\n🎉 ВСЕ ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ!")
            print("Все ID сохранены как в оригинальном JSON файле!")
            print("Изображения загружены в медиа-папку!")
        else:
            print("\n⚠️  ЗАГРУЖЕНО НЕ ВСЕ!")
            print("Пропущенные записи:")
            for detail in missing_details:
                print(f"  - {detail}")
    
    except Exception as e:
        print(f"❌ Произошла ошибка при загрузке: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
