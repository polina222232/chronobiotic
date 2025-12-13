from django.db import models
from django.db.models import Model
from django.urls import reverse


class Synonyms(models.Model):
    """
    Model representing a chem class (e.g. chronobiotics, radioprotectors).
    """
    synonymsmname = models.CharField(max_length=128, help_text="Enter a synonyms")
    originalbiotic = models.ForeignKey('Chronobiotic', on_delete=models.CASCADE, blank=True, related_name='synonyms')
    class Meta:
        db_table: str='synonyms'

    def __str__(self):
        """
        String for representing the Model object (in Admin site etc.)
        """
        return self.synonymsmname
class Articles(models.Model):
    """
    Model representing a chem class (e.g. chronobiotics, radioprotectors)
    """
    articlename = models.CharField(max_length=500, help_text="articlename")

    articleurl = models.URLField(max_length=250, help_text="Enter a link about this target")
    class Meta:
        db_table: str='article'

    def __str__(self):
        """
        String for representing the Model object (in Admin site etc.)
        """
        return self.articlename
class Targets(models.Model):
    """
    Model representing a chem class (e.g. chronobiotics, radioprotectors)
    """
    targetsname = models.CharField(max_length=150, help_text="Enter a target(Txxxxx)")
    targetsfullname=models.CharField(max_length=150, blank=True,help_text="Enter a fullname")
    targeturl = models.URLField(max_length=200, help_text="Enter a link about this target")
    class Meta:
        db_table: str='target'

    def __str__(self):
        """
        String for representing the Model object (in Admin site etc.)
        """
        return self.targetsname
class Effect(models.Model):
    """
    Model representing a chem class (e.g. chronobiotics, radioprotectors)
    """
    Effectname = models.CharField(max_length=150, help_text="Enter your effect")

    class Meta:
        db_table: str='effect'

    def __str__(self):
        """
        String for representing the Model object (in Admin site etc.)
        """
        return self.Effectname
class Mechanism(models.Model):
    """
    Model representing a chem class (e.g. chronobiotics, radioprotectors).
    """
    mechanismname = models.CharField(max_length=128, help_text="Enter a mechanism(e.g. relax, energy, tea , coffe)")
    class Meta:
        db_table: str='mechanism'

    def __str__(self):
        """
        String for representing the Model object (in Admin site etc.)
        """
        return self.mechanismname
class Bioclass(models.Model):
    """
    Model representing a chem class (e.g. chronobiotics, radioprotectors).
    """
    nameclass = models.CharField(max_length=128, help_text="Enter a class(e.g. chronobiotics, radioprotectors)")
    class Meta:
        db_table: str='class'

    def __str__(self):
        """
        String for representing the Model object (in Admin site etc.)
        """
        return self.nameclass

# class Clinicaltrials(models.Model):
#     """
#     Model representing a chem trials (e.g. humans,mouse,flyers).
#     """
#     trialsname = models.URLField(max_length=200, help_text="Enter a link about trials")
#     class Meta:
#         db_table: str='trials'
#
#     def __str__(self):
#         """
#         String for representing the Model object (in Admin site etc.)
#         """
#         return self.trialsname
class Chronobiotic(models.Model):

    gname = models.CharField(max_length=128, unique=True)
    smiles=models.CharField(max_length=256, unique=True)
    linkname = models.SlugField(max_length=256, null=True)
    molecula = models.CharField(max_length=256, unique=True)
    iupacname = models.CharField(max_length=256, unique=True)
    updphoto= models.ImageField(upload_to='media/')
    description = models.TextField(max_length=5000, help_text="Enter a  description of the biotic",blank=True)
    fdastatus = models.CharField(max_length=64,null=True,blank=True)
    articles= models.ManyToManyField(Articles, help_text="Select a targets of this ")
    linkslists=models.URLField(max_length=200,null=True,blank=True)
    pubchem = models.URLField(max_length=200,null=True,blank=True)
    chemspider = models.URLField(max_length=200, null=True,blank=True)
    drugbank = models.URLField(max_length=200, null=True,blank=True)
    chebi = models.URLField(max_length=200, null=True,blank=True)
    # chembil = models.URLField(max_length=200, null=True,blank=True)
    uniprot = models.URLField(max_length=200, null=True,blank=True)
    # engage = models.URLField(max_length=200, null=True,blank=True)
    kegg = models.URLField(max_length=200, null=True,blank=True)
    # msds = models.URLField(max_length=200, null=True,blank=True)
    # sider = models.URLField(max_length=200, null=True,blank=True)
    # toxnet = models.URLField(max_length=200, null=True,blank=True)
    selleckchem = models.URLField(max_length=200, null=True,blank=True)
    # isbn = models.CharField('ISBN',max_length=13, help_text='13 Character <a href="https://www.isbn-international.org/content/what-isbn">ISBN number</a>')
    classf = models.ManyToManyField(Bioclass, help_text="Select a class for this molecula")
    mechanisms = models.ManyToManyField(Mechanism, help_text="Select a mechanism of this ")
    target = models.ManyToManyField(Targets, help_text="Select a targets of this ")
    effect= models.ManyToManyField(Effect, help_text="Select a effects of this ", blank=True)
    # ManyToManyField used because genre can contain many books. Books can cover many genres.

    # clinictrials = models.ManyToManyField(Clinicaltrials, help_text="Select a class for this biotic")
    class Meta:
        db_table: str='chronobiotic'

    def __str__(self):
        """
        String for representing the Model object.
        """
        return self.gname
