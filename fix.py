import sys 
content = open('src/data/data_loader.py').read() 
old = 'label = 1 if (enzyme_overlap  or fda_score  else 0' 
new = 'import random\n        random.seed(hash(drug_a + drug_b) %% 1000)\n        label = 1 if enzyme_overlap  else random.randint(0, 1)' 
open('src/data/data_loader.py', 'w').write(content.replace(old, new)) 
print('Fixed') 
