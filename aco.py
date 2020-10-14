
# coding: utf-8

# In[1]:

from pandas_plink import read_plink
import numpy as np
import pandas as pd
from collections import OrderedDict as odict
from math import isnan
from scipy.stats import chi2_contingency
import timeit
import multiprocessing as mp
import os
import psutil
import pickle
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[97]:

class pre_processing:
    def __init__(self, plink_fn, pheno_fn, nbant, nbt, evaporation_rate, init_val, total_fitness_evals):
        self.pheno = self.read_pheno(pheno_fn)
        self.bim,self.fam,self.bed = read_plink(plink_fn)
        self.cases_i,self.controls_i = self.cases_controls()
        self.nbant = nbant
        self.nbt = nbt
        self.evaporation_rate = evaporation_rate
        self.init_val = init_val
        self.total_fitness_evals = total_fitness_evals
        
        
    
    def run(self):
        return self.bim, self.fam, self.bed, self.cases_i, self.controls_i, self.nbant, self.nbt, self.evaporation_rate, self.init_val, self.total_fitness_evals
        
    def cases_controls(self):
        cases_i = []
        controls_i =[]
        for index,rows in self.fam.iterrows():
            df = self.pheno.loc[self.pheno["FID"] == rows["fid"]]
            t2d = df.iat[0,4]
            if(t2d == '1'):
                cases_i.append(index)
            elif(t2d == '0'):
                controls_i.append(index) 
        return cases_i, controls_i
    

    def read_pheno(self,fn):
        header = odict(
            [
                ("FID", str),
                ("IID", str),
                ("AGE", str),
                ("BMI", str),
                ("T2D", str),
            ]
        )

        df = pd.read_csv(
            fn,
            delim_whitespace=True,
            header=None,
            names=header.keys(),
            dtype=header,
            compression=None,
            engine="c",
        )
        df["i"] = range(df.shape[0])
        return df
    
    
    


# In[98]:

class aco_gwas:
    def __init__(self, bim, fam, bed, cases_i, controls_i, nbant, nbt, evaporation_rate, init_val, total_fitness_evals):
        
        #Coppying the bim, bed and fam files
        self.bim = bim
        self.bed = bed
        self.fam = fam
        self.cases_i = cases_i
        self.controls_i = controls_i
        
        
        #Calculating the number of individuals and snps in the dataset
        self.n_individuals = self.fam.shape[0]
        self.n_snps = self.bim.shape[0]
        
        #Calculating the number of cases and controls
        self.n_cases = len(self.cases_i)
        self.n_controls = len(self.controls_i)
        
        #ACO setup
        self.nbant = nbant
        self.nbt = nbt
        self.init_val = init_val
        self.pheromone_mat = np.zeros(self.n_snps)
        self.evaporation_rate = evaporation_rate
        self.total_fitness_evals = total_fitness_evals
        self.numgen = total_fitness_evals/nbant
        self.best_p = 1
        self.best_stat = 0
        self.best_snp = []
        self.best_fitness = []
        
    
    def init_pheromone(self, init_val):
        self.pheromone_mat.fill(init_val)
    
    def tournament_choice(self):
        tournament = np.random.randint(self.n_snps, size = nbt)
        biggest = 0
        biggest_i = 0
        for i in tournament:
            if(self.pheromone_mat[i] > biggest):
                biggest = self.pheromone_mat[i]
                biggest_i = i
        return biggest_i
                
            
    def chi_sq_omnibus(self, snp1, snp2):
        snp1_mat = self.bed[snp1].compute()
        snp2_mat = self.bed[snp2].compute()
        
        t_cases = np.zeros((4,4))
        t_controls = np.zeros((4,4))
        table_i = []
        table_j = []
        
        for i in self.cases_i:
            if(isnan(snp1_mat[i])):
                snp1_mat[i] = 3
                
            if(isnan(snp2_mat[i])):
                snp2_mat[i] = 3  
                
            t_cases[int(snp1_mat[i])][int(snp2_mat[i])] += 1
            
            
        for i in self.controls_i:
            
            if(isnan(snp1_mat[i])):
                snp1_mat[i] = 3
                
            if(isnan(snp2_mat[i])):
                snp2_mat[i] = 3
                
            t_controls[int(snp1_mat[i])][int(snp2_mat[i])] += 1
            
        
        table_i = t_cases[:3,:3]
        table_i = table_i.flatten()
        non_zeros = np.nonzero(table_i)
        table_i = table_i[non_zeros]
        
        table_j = t_controls[:3,:3]
        table_j = table_j.flatten()
        table_j = table_j[non_zeros]
        
        table = [table_i, table_j]
        stat, p, dof, expected = chi2_contingency(table)
        
        return stat, p 
    
    def update_pheromone(self, snp1, snp2):
        chi_sq_val, p_val = self.chi_sq_omnibus(snp1,snp2)
        if(p_val < self.best_p):
            self.best_stat = chi_sq_val
            self.best_p = p_val
            self.best_snp = [snp1,snp2]
        self.pheromone_mat[int(snp1)] += chi_sq_val
        self.pheromone_mat[int(snp2)] += chi_sq_val
        
        
    def evaporate_pheromone(self):
        self.pheromone_mat *= self.evaporation_rate
        
        
    def ant(self):
        snp1 = self.tournament_choice()
        snp2 = self.tournament_choice()
        self.update_pheromone(snp1,snp2)
        
    
    def run(self):
        #pool = mp.Pool(processes = 3)
        self.init_pheromone(self.init_val)
        
        for i in range(self.numgen):
            for j in range(nbant):
                self.ant()
            #pool.map(self.ant, [row for row in range(nbant)])  
            self.best_fitness.append(np.amax(self.pheromone_mat))
            self.evaporate_pheromone()  
        
        return self.pheromone_mat,self.best_fitness, self.best_p        


# In[ ]:

class post_processing:
    def __init__(self,bim, fam):
        self.bim = bim
        self.fam = fam
    
    def save_file(self, fn, val):
        fileObject = open(fn,'wb')
        pickle.dump(val,fileObject)   
        fileObject.close()

    
    def load_file(self, fn):
        fileObject = open(fn,'rb')
        val = pickle.load(fileObject)
        fileObject.close()
        return val
    
    def best_snps(self, n_best_snps, p_values):
        top_snps = np.argsort(p_values)[:n_best_snps]
        for i in top_snps:
            print("SNP: ", bim.loc[i, 'snp'], "P-value: ", p_values[i])
    
    def manhatten_plot(self, p_values,fn):
        self.bim['minuslog10pvalue'] = -np.log10(pvalues)
        m_plot = self.bim[['i','chrom','minuslog10pvalue' ]]
        df_grouped = m_plot.groupby(('chrom'))
        plt.style.use('ggplot')
        fig = plt.figure(dpi = 300)
        ax = fig.add_subplot(111)
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
        x_labels = []
        x_labels_pos = []

        for num, (name, group) in enumerate(df_grouped):
            group.plot(kind='scatter', x='i', y='minuslog10pvalue',color=colors[num % len(colors)], ax=ax, s = 10000000/len(bim))
            x_labels.append(name)
            x_labels_pos.append((group['i'].iloc[-1] - (group['i'].iloc[-1] - group['i'].iloc[0])/2))
        ax.set_xticks(x_labels_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlim([0, len(m_plot)])
        ax.set_ylim([0, 30])
        ax.set_xlabel('Chromosome')
        plt.xticks(fontsize=6, rotation=60)
        plt.yticks(fontsize=6)
        plt.savefig(fn, bbox_inches='tight')        


if(__name__ == '__main__'):
    plink_fn = '/Users/raouldias/Desktop/Extend/extend_csp_data_annon'
    pheno_fn = '/Users/raouldias/Desktop/Extend/extend_phenotype.txt'
    nbant = 200
    nbt = 1000
    evaporation_rate = 0.99
    init_val = 1
    total_fitness_evals = 1000000

    bim, fam, bed, cases_i, controls_i, nbant, nbt, evaporation_rate, init_val, total_fitness_evals = pre_processing(plink_fn, pheno_fn, nbant, nbt, evaporation_rate, init_val, total_fitness_evals).run()
    aco = aco_gwas(bim, fam, bed, cases_i, controls_i, nbant, nbt, evaporation_rate, init_val, total_fitness_evals)




    pheromone_matrix, best_fitness, best_p = aco1.run()






