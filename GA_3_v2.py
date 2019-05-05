def mutation():
    global x,PS,mutation_num
    mutation_count =0
    for i in range(PS+taguchi_num):
        probability = np.random.uniform()
        if (probability <= mutationRate):
            mutation_num +=1
            chosen_num = np.random.randint(0,8) #chose gene
            chosen_cro = np.random.randint(0,PS+taguchi_num) #chose cromosome
            x[PS+taguchi_num+mutation_count] = x[i]
            x[PS+taguchi_num+mutation_count,chosen_num] = (x[i,chosen_num] + x[chosen_cro,chosen_num])*0.5
            
            mutation_count+=1
            
def taguchi_method(x_tmp):
    global fit_cro,taguchi_fitSum,taguchi_xNum,taguchi_num
    x_result = np.zeros([int(taguchi_num),gene])
    fit_result = np.zeros([int(taguchi_num)])
    
    for i in range(0, taguchi_xNum, 2):
        x_cro = np.zeros([2,gene])
        x_taguchi = np.zeros([16,gene])
        fit_cro = np.zeros([16,2])
        x_cro[:] = x_tmp[i:i+2]
        
        for j in range(16): #16 experiments
            for k in range(gene):
                x_taguchi[j,k] = x_cro[int(taguchi_chart[j,k]) - 1, k]
        fit_tmp = cal_fitness(x_taguchi, 16) #fitness of 16 experiments
        fit_cro[:16,0] = fit_tmp[:16]
        fit_cro[:16,1] = np.log(1/(fit_cro[:16,0]*fit_cro[:16,0]))
        
        #compaire
#        taguchi_fitSum = np.zeros([3,8])
        taguchi_fitSum[2] = -10000
        x_new = np.zeros([1,gene])
        
        for m in range(8): #levels for 8 gene
            for n in range(16): #indices of levels
                if(taguchi_chart[n,m] == 1):
                    taguchi_fitSum[0,m] += fit_cro[n,1]
                else:
                    taguchi_fitSum[1,m] += fit_cro[n,1]
            if (taguchi_fitSum[0,m] > taguchi_fitSum[1,m]):
                taguchi_fitSum[2,m] = 0
            else:   
                taguchi_fitSum[2,m] = 1
            x_new[0,m] = x_cro[int(taguchi_fitSum[2,m]),m]

        fit_new = cal_fitness(x_new, 1)
        if(fit_new < np.min(fit_tmp)):
            x_result[i//2] = x_new[0]
            fit_result[i//2] = fit_new
        else:
            index = np.argmin(fit_tmp)
            x_result[i//2] = x_taguchi[index]
            fit_result[i//2] = fit_tmp[index]
            
    return x_result,fit_result

def crossover():
    global x,fit,x_min,taguchi_num,taguchi_xNum

    reproduction_probability = chosen_probability()
    x_tmp = np.zeros([taguchi_xNum, gene])
    
    x_tmp[0] = x_min
    for i in range(1,PS): #select cromosomes for crossover
        probability = np.random.uniform()
        for j in range(PS): #reproduction of probability
            if (probability <= reproduction_probability[j]):
                x_tmp[i] = x[j]
                break
    
    x_result,fit_result = taguchi_method(x_tmp)
    x[PS:PS+len(x_result)] = x_result
    fit[PS:PS+len(fit_result)] = fit_result
    
def chosen_probability():
    global fit,PS
    
    fit_PS = fit[:PS]
    reciprocal = np.reciprocal(fit_PS) #對每一fit取倒數

    reciprocalSum = np.sum(reciprocal, axis=0) #累加倒數
    chosen_probability = reciprocal/reciprocalSum #計算選擇機率
    reproduction_probability = chosen_probability.cumsum(axis=0) #累加選擇機率
    return reproduction_probability

def reproduction():
    global funcall,x,fit
    funcall += 1
    
    reproduction_probability = chosen_probability()
    x_tmp = np.copy(x)
    fit_tmp = np.copy(fit)
    
    for i in range(1, PS):
        probability = np.random.uniform()
        for j in range(PS): #reproduction of probability
            if (probability <= reproduction_probability[j]):
                x[i] = x_tmp[j]
                fit[i] = fit_tmp[j]
                break

def sorting():
    global fit,x
    sort = np.argsort(fit) #indices list
    fit = np.sort(fit)
    x_tmp = np.zeros([exPS, gene])

    for i in range(exPS):
        x_tmp[i] = x[sort[i]]
        
    x = np.copy(x_tmp)
    
def cal_fitness(x, num):
    global w
    fit_tmp = np.zeros([num])
    for i in range(num):
        unfeasibility = 0 #unfeasibility
        c = np.zeros([6]) #memorize calculation of constraint
        c[0] = 1 - 0.0025 * (x[i,3] + x[i,5])
        c[1] = 1 - 0.0025 * (x[i,4] + x[i,6] - x[i,3])
        c[2] = 1 - 0.01 * (x[i,7] - x[i,4])     
        c[3] = x[i,0] * x[i,5] - 833.33252 * x[i,3] -100 * x[i,0] + 83333.333
        c[4] = x[i,1] * x[i,6] - 1250 * x[i,4] - x[i,1] * x[i,3] + 1250 * x[i,3]
        c[5] = x[i,2] * x[i,7] - 1250000 - x[i,2] * x[i,4] + 2500 * x[i,4]
        for j in range(6):
            if c[j] < 0:
                unfeasibility += c[j]
        unfeasibility = abs(unfeasibility)
        p = w * unfeasibility #penalty
        fit_tmp[i] = x[i,0] + x[i,1] + x[i,2] + p
        
    return fit_tmp

def generate_Chromosomes():
    x_1  = np.random.uniform(100,10000,1)
    x_2_3 = np.random.uniform(1000,10000,2)
    x_4_8 = np.random.uniform(10,1000,5)
    x_com = np.concatenate((x_1,x_2_3,x_4_8)) #combine genes
    x_com = np.around(x_com,decimals = 6)
    return x_com

def initialization():
    for i in range(PS):
        x[i] = generate_Chromosomes()

import numpy as np
from numpy import genfromtxt
taguchi_chart = genfromtxt('123.csv', delimiter=',') 
PS = 200                           #population size
exPS = PS*3                        #extra population
gene = 8                           #gene number
w = 100000                         #penalty
#corssoverRate = 0.5
taguchi_xNum = PS*2
taguchi_num = taguchi_xNum//2
mutationRate = 0.1
mutation_num=0
iteration = 1000
keepRate = 0.8
funcall = 0                        #number of function call
global_min = 100000000.0

x = np.full((exPS, gene), 10000.0) #cromosome
fit = np.full((exPS),10000000.0)   #fitness
x_min = np.zeros([gene])           #best solution
fit_cro = np.zeros([16,2])         #crossober
taguchi_fitSum = np.zeros([3,8])   #taguchi method

''' main '''
initialization()
fit[:PS] = cal_fitness(x, PS)
sorting()

while(iteration > 0):
    if (iteration<900):
        keepRate = 0.02
        
    iteration -= 1
    reproduction()
    crossover()
    mutation()
    fit = cal_fitness(x, exPS)
    sorting()

    for i in range(int(PS*keepRate), PS): #generate random cromosome untill 200 cromosomes
        x[i] = generate_Chromosomes()
    x[PS:] = 10000.0
    fit[PS:] = 10000000.0
        
    tmp_min = np.min(fit)
    
    print('iter = ',iteration,tmp_min)
    
    if(tmp_min<global_min):     #record best solution
        global_min = tmp_min
        x_min = x[0]
    else:
        x[2] = x_min
    
print(global_min)
#answer 7049
#mutation method