from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from .models import Chronobiotic
from django.http import JsonResponse
from django.db.models import Q


def index(request):
    query = request.GET.get('search', '')
    model = Chronobiotic.objects.all().prefetch_related(
        'effect',
        'target',
        'mechanisms',
        'articles',
        'synonyms'  # Добавляем предзагрузку синонимов
    )

    if query:
        model = model.filter(
            Q(gname__icontains=query) |
            Q(molecula__icontains=query) |
            Q(fdastatus__icontains=query) |
            Q(smiles__icontains=query) |
            Q(effect__Effectname__icontains=query) |
            Q(target__targetsname__icontains=query) |
            Q(mechanisms__mechanismname__icontains=query) |
            Q(articles__articlename__icontains=query) |
            Q(synonyms__synonymsmname__icontains=query)  # Добавляем поиск по синонимам
        ).distinct()
    else:
        model = Chronobiotic.objects.all()

    # Пагинация
    paginator = Paginator(model, 25)  # По 25 записей на страницу
    page_number = request.GET.get('page')  # Получаем текущую страницу из запроса
    page_obj = paginator.get_page(page_number)  # Получаем объект страницы

    return render(request, 'main/index.html', {'page_obj': page_obj, 'query': query})

def about(request):
    return render(request, 'main/about.html')
def substance_detail(request, linkname):
    substance = get_object_or_404(Chronobiotic, linkname=linkname)
    return render(request, 'main/substance_detail.html', {'substance': substance})
def get_synonyms(request, linkname):
    try:
        chronobiotic = Chronobiotic.objects.get(linkname=linkname)
        synonyms = list(chronobiotic.synonyms.all().values_list('synonymsmname', flat=True))
        return JsonResponse({'synonyms': synonyms})
    except Chronobiotic.DoesNotExist:
        return JsonResponse({'synonyms': []})
