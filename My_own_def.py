import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from tqdm.auto import tqdm
from scipy.stats import mannwhitneyu

plt.style.use('ggplot')

def activ(data):        
    df = data.reset_index()
    activ=[]
    for i in range(len(df)):
        if df.loc[i:i]['activ'].all()>0:
            activ.append('on')
        else:
            activ.append('off')
    df['activ']=activ
    return df

def checks(data):        
    df = data.reset_index()
    checks=[]
    for i in range(len(df)):
        if df.loc[i:i]['checks'].all()>0:
            checks.append(1)
        else:
            checks.append(0)
    df['checks']=checks
    return df

def ARPU_ARPPU(data):
    stat = data.groupby('group').agg({'rev':['mean','min','max','count','sum'],'group':'count'})
    A_mean=stat['rev']['mean']['A']
    B_mean=stat['rev']['mean']['B']
    A_min=stat['rev']['min']['A']
    B_min=stat['rev']['min']['B']
    A_max=stat['rev']['max']['A']
    B_max=stat['rev']['max']['B']
    A_count=stat['rev']['count']['A']
    B_count=stat['rev']['count']['B']
    A_sum=stat['rev']['sum']['A']
    B_sum=stat['rev']['sum']['B']
    all_A=data.group.value_counts().A
    all_B=data.group.value_counts().B
    APRU_A=round(A_sum/all_A,2)
    APRU_B=round(B_sum/all_B,2)
    ARPPU_A=round(A_sum/A_count,2)
    ARPPU_B=round(B_sum/B_count,2)
    df = pd.DataFrame({
    'ARPU': [APRU_A, APRU_B],
    'ARPPU': [ARPPU_A, ARPPU_B]
    })
    print(df)

def data_info(data):
    colum_name=data.columns.values.tolist ()
    print('')
    print('Проверяем таблицу по столбцу',colum_name[0])
    for x in data.duplicated([colum_name[0]]):
        if x ==True:
            print('Есть дубликаты в первом столбце, стоит их учесть')
    print(data.shape[0],'Записей')
    print(data.shape[1],'Столбцов/а')
    nul=data[[colum_name[0]]].isnull().sum()
    if nul.all()>0:
        print('Есть нуль значения')
    else:
        print('Нету нуль значений')
    na=data[[colum_name[0]]].isna().sum()
    if na.all()>0:
        print('Есть nan значения')
    else:
        print('Нету nan значений')
    
def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1), 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            len(data_column_1), 
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1)-statistic(samples_2)) # mean() - применяем статистику
        
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}

def revenue(data):        
    df = data.reset_index()
    revenue=[]
    for i in range(len(df)):
        if df.loc[i:i]['rev'].all()>0:
            revenue.append(1)
        else:
            revenue.append(0)
    df['revenue']=revenue
    return df.groupby('revenue').count()

