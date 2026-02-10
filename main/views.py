import json
import time

from django.core.paginator import Paginator
from django.db.models import Q
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import Chronobiotic


def index(request):
    query = request.GET.get('search', '')
    model = Chronobiotic.objects.all().prefetch_related('effect', 'target', 'mechanisms', 'articles', 'synonyms')
    
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
            Q(synonyms__synonymsmname__icontains=query)
        ).distinct()
    
    paginator = Paginator(model, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
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


def agent_chat(request):
    return render(request, 'main/agent_chat/agent_chat.html')


@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        
        # Simple response generation
        msg_lower = message.lower()
        
        if 'chronobiotic' in msg_lower:
            response = """**Chronobiotics** are pharmacological agents that modify circadian rhythm parameters.

**Main classes:**
- Natural chrononutrients (melatonin, polyphenols)
- Synthetic modulators (KL001, KS15)
- Chronobiotic hormones (ramelteon, tasimelteon)

**Applications:** Sleep disorders, jet lag, metabolic diseases."""
        
        elif 'melatonin' in msg_lower:
            response = """**Melatonin** is the best-known chronobiotic hormone.

**Functions:**
- Regulates sleep-wake cycle
- Acts on MT1/MT2 receptors
- Produced by pineal gland in darkness

**Uses:** Insomnia, jet lag, circadian rhythm disorders"""
        
        elif 'kl001' in msg_lower or 'ks15' in msg_lower:
            response = """**KL001 and KS15** are synthetic chronobiotics.

- **KL001:** CRY stabilizer, lengthens circadian period
- **KS15:** CRY activator, modifies rhythm parameters

*Source: Solovev et al. (2021) Clocks & Sleep*"""
        
        elif 'target' in msg_lower or 'clock' in msg_lower:
            response = """**Molecular Targets of Chronobiotics:**

- **CLOCK/BMAL1** - Core clock transcription factors
- **PER/CRY** - Negative regulators
- **ROR/REV-ERB** - Nuclear receptors
- **MT1/MT2** - Melatonin receptors"""
        
        else:
            response = """I'm ChronobioticsAI! I can help with:

🔬 **Compounds** - Melatonin, KL001, KS15, Ramelteon
🎯 **Targets** - CLOCK, BMAL1, CRY, PER, MT1/MT2
📚 **Research** - Articles, mechanisms, FDA status

Try asking:
- "What are chronobiotics?"
- "Tell me about melatonin"
- "How do KL001 and KS15 work?"""
        
        citations = []
        if 'chronobiotic' in msg_lower:
            citations.append({
                'authors': ['Solovev, I. A.', 'Golubev, D. A.'],
                'title': 'Chronobiotics classifications',
                'journal': 'Biomeditsinskaya Khimiya',
                'year': 2024
            })
        
        return JsonResponse({'success': True, 'response': response, 'citations': citations})
    
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def chat_stream(request):
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        
        msg_lower = message.lower()
        
        if 'chronobiotic' in msg_lower:
            response = "Chronobiotics are pharmacological agents that modify circadian rhythm parameters. "
            response += "They include natural compounds like melatonin, synthetic modulators like KL001 and KS15, "
            response += "and drugs like ramelteon and tasimelteon. They are used for sleep disorders, jet lag, and circadian rhythm disorders."
        
        elif 'melatonin' in msg_lower:
            response = "Melatonin is a hormone produced by the pineal gland that regulates sleep-wake cycles. "
            response += "It acts on MT1 and MT2 receptors in the brain's suprachiasmatic nucleus. "
            response += "It's commonly used for insomnia, jet lag, and circadian rhythm disorders."
        
        else:
            response = "I'm ChronobioticsAI! I can help you learn about chronobiotic compounds, their molecular targets, "
            response += "FDA approval status, and research articles. Try asking about melatonin, KL001, or chronobiotics classification."
        
        def generate():
            words = response.split()
            for i, word in enumerate(words):
                chunk = word + (' ' if i < len(words) - 1 else '')
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                time.sleep(0.03)
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        return StreamingHttpResponse(generate(), content_type='text/event-stream')
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def search_database(request):
    try:
        data = json.loads(request.body)
        query = data.get('query', '')
        results = []
        
        if query:
            substances = Chronobiotic.objects.filter(
                Q(gname__icontains=query) |
                Q(description__icontains=query)
            )[:10]
            
            for sub in substances:
                results.append({
                    'type': 'compound',
                    'title': sub.gname,
                    'description': sub.description[:200] if sub.description else '',
                    'url': f'/substance/{sub.linkname}/'
                })
        
        return JsonResponse({'success': True, 'results': results})
    
    except Exception as e:
        return JsonResponse({'error': str(e), 'success': False}, status=500)
