from django.contrib import admin
from .models import  Chronobiotic, Targets , Bioclass, Mechanism,Synonyms, Effect,Articles

@admin.register(Chronobiotic)
class Chronobioadm(admin.ModelAdmin):
    prepopulated_fields = {'linkname':('gname',)}
#admin.site.register(Chronobiotic)
# admin.site.register(Clinicaltrials)
admin.site.register(Bioclass)
admin.site.register(Mechanism)
admin.site.register(Synonyms)
admin.site.register(Targets)
admin.site.register(Effect)
admin.site.register(Articles)