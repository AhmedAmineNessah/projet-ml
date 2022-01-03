#!/usr/bin/env python
# coding: utf-8

# # Membres
# 
# Ahmed Farjallah
# 
# Ahmed Amine Nessah
# 
# Mehdi Taieb Amri
# 
# Aziz Maatouk
# 
# Mouadh Belgaied
# 
# Youssef Ben Amara
# 
# Ahmed Shili

# In[179]:


# import librairises
#get_ipython().system('pip install catboost')
#get_ipython().system('pip install xgboost')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from catboost import CatBoostClassifier
from catboost import Pool
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# ## Variables

# #### ID: ID of each client
# #### LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# #### SEX: Gender (1=male, 2=female)
# #### EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# #### MARRIAGE: Marital status (1=married, 2=single, 3=others)
# #### AGE: Age in years
# #### PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
# #### PAY_2: Repayment status in August, 2005 (scale same as above)
# #### PAY_3: Repayment status in July, 2005 (scale same as above)
# #### PAY_4: Repayment status in June, 2005 (scale same as above)
# #### PAY_5: Repayment status in May, 2005 (scale same as above)
# #### PAY_6: Repayment status in April, 2005 (scale same as above)
# #### BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# #### BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# #### BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# #### BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# #### BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# #### BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# #### PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# #### PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# #### PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# #### PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# #### PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# #### PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# #### default.payment.next.month: Default payment (1=yes, 0=no)

# # Introduction
# 
# En regardant le problème, nous voyons une utilisation potentielle de ce type de données : dans quelle mesure pouvons-nous prédire, mois par mois, le défaut de nos clients ? En d'autres termes, quelle est la performance de notre modèle si nous utilisons uniquement les données des 2 premiers mois par rapport au moment où nous utilisons 6 mois d'historique de paiement ?
# 
# Cependant, comme nous avons beaucoup de chemin à parcourir, nous allons nous concentrer sur un problème plus simple : peut-on prévoir le défaut avec un mois d'avance ?
# 
# Le cahier est structuré comme suit :
# 
# Première exploration : juste pour voir ce que nous avons.
# Nettoyage : il est temps de faire des modifications sur les variables non etiquettées
# Résultat final et leçons apprises

# # Data exploration

# In[180]:


# import the datasets and rename columns

data = pd.read_excel('default of credit card clients.xls')
data = data.drop("Unnamed: 0", axis = 1)
data = data.drop(index = 0, axis = 0)
data.info()


# In[181]:


data = data.rename(columns={'X1':'LIMIT_BAL', 'X2':'SEX', 'X3':'EDUCATION', 'X4':'MARRIAGE', 'X5':'AGE', 'X6':'PAY_1', 'X7':'PAY_2',
       'X8':'PAY_3', 'X9':'PAY_4', 'X10':'PAY_5', 'X11':'PAY_6', 'X12':'BILL_AMT1', 'X13':'BILL_AMT2',
       'X14':'BILL_AMT3', 'X15':'BILL_AMT4', 'X16':'BILL_AMT5', 'X17':'BILL_AMT6', 'X18':'PAY_AMT1',
       'X19':'PAY_AMT2', 'X20':'PAY_AMT3', 'X21':'PAY_AMT4', 'X22':'PAY_AMT5', 'X23':'PAY_AMT6', 'Y':'def_pay'})
data.head()


# In[182]:


# hape
data.shape


# In[183]:


data.info()


# In[184]:


# tranforme tye of column for  the dataset
data = data.astype(int)
data.info()


# In[185]:


data.describe()


# ### interpretation 

# ###### D'apres la describtion de la dataset on constate que:
#     1-le nombres des femmes est supperieur > Hommes( mean(sex)=1.603)
#     2-la moyenne d'age est de 35ans d'ou on peut dire que la majorité des personnes ont terminé leurs parcours universitaire
#     3-le nombre minimal de Pay_M =-2 qui n'existe pas dans la documentation
#     4-50% des personnes sont singles
#     5-min(BillAmt_m) est negative qui n'est pas logique il faut l'etudier

# ## Default

# In[186]:


data.def_pay.value_counts()


# In[187]:


sns.countplot(x='def_pay', data=data)
plt.show()


# In[188]:


def_cnt = (data.def_pay.value_counts(normalize=True)*100)
def_cnt.plot.bar(figsize=(6,6))
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.title("Probabilité de Default de paiement du prochain mois", fontsize=15)
for x,y in zip([0,1],def_cnt):
    plt.text(x,y,y,fontsize=12)
plt.show()


# ### Categorical Variables

# ### SEX

# In[189]:


# SEX
Sex = data['SEX'].value_counts()
Sex = (Sex/len(data))*100
Sex


# ###### on constate que il existe nombre femme > nombre homme

# In[190]:


labels = ['femme','homme']
plt.pie(Sex,labels=labels,autopct='%.0f%%')
plt.show()


# In[191]:


plt.figure(figsize=(12,4))

ax = sns.countplot(data = data, x = 'SEX', hue="def_pay")

plt.xlabel("Sex", fontsize= 12)
plt.ylabel("nombre des Clients", fontsize= 12)
plt.ylim(0,20000)
plt.xticks([0,1],['Male', 'Female'], fontsize = 11)

for p in ax.patches:
    ax.annotate((p.get_height()), (p.get_x()+0.16, p.get_height()+1000))

plt.show()


# In[192]:


data[["SEX", "def_pay"]].groupby(['SEX']).mean().sort_values(by='def_pay')


# ###### on constate qu'il y a beaucoup plus de femmes que d'hommes ,de plus les hommes ont une chance légèrement plus élevée de défaut de paiement. Nous essaierons de le confirmer un peu plus loin en comparant des hommes et des femmes de même niveau d'éducation et de même état matrimonial.

# ### Marriage

# In[193]:


# Marriage
Marriage = data['MARRIAGE'].value_counts()
MarriageDef = data.groupby(['MARRIAGE', 'def_pay']).size().unstack(1)
MarriageDef


# ###### on constate que il existe 0 qui n'est pas mentionner dans la documentation 
# ###### il faut regrouper avec others

# In[194]:


MarriageDef.plot(kind = 'bar')


# In[195]:


labels = ['single','married','others','unknown']
Marriage=(Marriage/len(data))*100
Marriage
plt.pie(Marriage,labels=labels,autopct='%.0f%%')
plt.show()


# ### Education

# In[196]:


# Eductation
EducationDef = data.groupby(['EDUCATION', 'def_pay']).size().unstack(1)
Education = data['EDUCATION'].value_counts()
EducationDef


# In[197]:


EducationDef.plot(kind = "barh")


# In[198]:


data['def_pay'].groupby(data['EDUCATION']).value_counts(normalize = True)


# In[199]:


counts_left=[10585,14030,4917,123,345]
department_left=['graduate school','university','high school','others','UNkown']
explode=[0,0.1,0.1,0.1,0.1]
fig1, ax1 = plt.subplots(figsize=(15,15))
ax1.pie(counts_left, explode=explode,labels=department_left, autopct='%1.1f%%',
        shadow=True, startangle=150)
ax1.axis('equal')  
plt.show()


# In[200]:


plt.figure(figsize=(12,4))

ax = sns.countplot(data = data, x = 'EDUCATION', hue="def_pay")

plt.xlabel("Education", fontsize= 12)
plt.ylabel("nombre client of Clients", fontsize= 12)
plt.ylim(0,12000)
plt.xticks([0,1,2,3,4],['Grad School','University','High School','Others','Unknown'], fontsize = 11)

for p in ax.patches:
    ax.annotate((p.get_height()), (p.get_x()+0.11, p.get_height()+500))

plt.show()


# In[201]:


plt.figure(figsize=(12,4))

ax = sns.barplot(x = "EDUCATION", y = "def_pay", data = data, ci = None)

plt.ylabel("% of Default", fontsize= 12)
plt.ylim(0,0.5)
plt.xticks([0,1,2,3,4],['Grad School','University','High School','Others','Unknown'], fontsize = 11)

for p in ax.patches:
    ax.annotate("%.2f" %(p.get_height()), (p.get_x()+0.30, p.get_height()+0.03),fontsize=13)

plt.show()


# ### Le niveau d'éducation prédominant dans notre ensemble de données est « Université », suivi par « École supérieure », « Lycée », « Inconnu » et « Autres ».
# 
# ### En ne considérant que les trois premiers niveaux, il semble qu'une éducation supérieure se traduise par une moindre chance d'échec. Cependant, à la fois « Inconnu » et « Autres » (dont nous présumons que cela signifie un niveau inférieur à celui du secondaire), ont une probabilité sensiblement plus faible.

# ### Conclusion (Categorical Variable)
# ### d'apres les graphes et les tableaux on deduit que edcutation et mariage doivent etre netoyer (eliminer les elment qui ne sont pas mentionner dans lle document comme le 0 pour le columns education )

# ### (LIMIT_BAL) + Demographic Features

# In[202]:


plt.figure(figsize=(12,6))

sns.boxplot(x = "SEX", y = "LIMIT_BAL",data = data, showmeans=True, 
            meanprops={"markerfacecolor":"red",  "markeredgecolor":"black", "markersize":"10"})

plt.ticklabel_format(style='plain', axis='y') #repressing scientific notation    
plt.xticks([0,1],['Male', 'Female'], fontsize = 12)

plt.show()


# In[203]:


plt.figure(figsize=(14,6))

sns.boxplot(x = "EDUCATION", y = "LIMIT_BAL", data = data,  showmeans=True, 
            meanprops={"markerfacecolor":"red",  "markeredgecolor":"black", "markersize":"10"})

plt.ticklabel_format(style='plain', axis='y') #repressing scientific notation   
plt.xticks([0,1,2,3,4],['Grad School','University','High School','Others','Unknown'], fontsize = 11)

plt.show()


# In[204]:


plt.figure(figsize=(14,6))

sns.boxplot(x = "MARRIAGE", y = "LIMIT_BAL", data = data,showmeans=True, 
            meanprops={"markerfacecolor":"red",  "markeredgecolor":"black", "markersize":"10"})

plt.ticklabel_format(style='plain', axis='y') #repressing scientific notation    
plt.xticks([0,1,2,3],['Unknown', 'Married', 'Single', 'Divorce'], fontsize = 11)

plt.show()


# ### Numerical Variables

# In[205]:


# draw_histograms permet de visualiser la distrubtion des variables 
def draw_histograms(df, variables, n_rows, n_cols, n_bins):
    fig=plt.figure()
    fig.set_size_inches(18,8)
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=n_bins,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()


# In[206]:


sns.pairplot(data, vars=data.columns[11:17],hue= 'def_pay')


# In[207]:


g = sns.FacetGrid(data,hue='def_pay')
g.map(plt.hist, 'BILL_AMT1', alpha=0.9, bins=25) 
g.add_legend()
g2 = sns.FacetGrid(data,hue='def_pay')
g2.map(plt.hist, 'BILL_AMT2', alpha=0.9, bins=25) 
g2.add_legend()
g3 = sns.FacetGrid(data,hue='def_pay')
g3.map(plt.hist, 'BILL_AMT3', alpha=0.9, bins=25) 
g3.add_legend()
g4 = sns.FacetGrid(data,hue='def_pay')
g4.map(plt.hist, 'BILL_AMT4', alpha=0.9, bins=25) 
g4.add_legend()
g5 = sns.FacetGrid(data,hue='def_pay')
g5.map(plt.hist, 'BILL_AMT5', alpha=0.9, bins=25) 
g5.add_legend()
g6 = sns.FacetGrid(data,hue='def_pay')
g6.map(plt.hist, 'BILL_AMT6', alpha=0.9, bins=25) 
g6.add_legend()


# In[208]:


billsDef = data.groupby(['BILL_AMT1', 'def_pay']).size().unstack(1)

#bills = data[['BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
#draw_histograms(billsDef, billsDef.columns , 2, 3, 20)


# In[209]:


bills = data[['BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
draw_histograms(bills, bills.columns, 2, 3,20)


# In[210]:


pay = data[['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
draw_histograms(pay, pay.columns, 2, 3, 20)


# ######  L'histogramme ci-dessus montre la distribution du montant de la facture généré pour chaque mois explicitement pour les défaillants et les non défaillants

# In[211]:


pays = data[['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
draw_histograms(pays, pays.columns, 2, 3, 20)


# ###### La figure ci-dessus montre un graphique à barres pour chaque statut de paiement mensuel qui indique le nombre de défaillants et de non défaillants.

# ###### Conclusion : 
#     1 - La majorité des personnes ont Bill_amout == 0
#     2 - Il existe des valeurs negatives de Bill_amount : On peut les cosidérés comme payés avec excèes
#     3 - D'apres les trois histogramme : on trouve 0 et -2 dans les PAY_n => donc on peut les regroupés comme -1 : payé

# # Data Cleaning

# ### *1/ Categorical Variable*

# On regroupe 5,6,0 dans une catégorie ohter = 4

# In[212]:


fil = (data.EDUCATION == 5) | (data.EDUCATION == 6) | (data.EDUCATION == 0)
data.loc[fil, 'EDUCATION'] = 4
data.EDUCATION.value_counts()


# on regroupe 0 dans other = 3

# In[213]:


data.loc[data.MARRIAGE == 0, 'MARRIAGE'] = 3
data.MARRIAGE.value_counts()


# ### *2/ Numerical Variable* 

# on regroupe -2,-1 dans 0 qui represente le pay duly

# In[214]:


fil = (data.PAY_1 == -2) | (data.PAY_1 == -1) | (data.PAY_1 == 0)
data.loc[fil, 'PAY_1'] = 0
fil = (data.PAY_2 == -2) | (data.PAY_2 == -1) | (data.PAY_2 == 0)
data.loc[fil, 'PAY_2'] = 0
fil = (data.PAY_3 == -2) | (data.PAY_3 == -1) | (data.PAY_3 == 0)
data.loc[fil, 'PAY_3'] = 0
fil = (data.PAY_4 == -2) | (data.PAY_4 == -1) | (data.PAY_4 == 0)
data.loc[fil, 'PAY_4'] = 0
fil = (data.PAY_5 == -2) | (data.PAY_5 == -1) | (data.PAY_5 == 0)
data.loc[fil, 'PAY_5'] = 0
fil = (data.PAY_6 == -2) | (data.PAY_6 == -1) | (data.PAY_6 == 0)
data.loc[fil, 'PAY_6'] = 0
late = data[['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
draw_histograms(late, late.columns, 2, 3, 10)


# on constate que ceux qui ont des cartes de credit payent maximum apres 2 mois , donc on peut travailler sueleument avec 
# PAY_1 et PAY_2

# In[215]:


#Hypothesis confirmed
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), annot=True)


# # Normalisation

# In[216]:


sc = StandardScaler()
X  = data.drop('def_pay',axis =1)
X_2  = data.drop('def_pay',axis =1)
Y=data['def_pay']
X_sc = sc.fit_transform(X)
X=pd.DataFrame(X_sc,columns=X.columns)
# = sc.fit_transform(data['BILL_AMT1'].values.reshape(-1,1))
#model = PCA(.95)
#X_reduce = model.fit_transform(X)
#X_reduce


# In[217]:


#plt.plot(np.cumsum(model.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance')
#plt.grid(True)
#plt.show()


# ### KNeighborsClassifier Algorithm 

# In[218]:


#X = data.drop(['def_pay'],axis=1)
X_train,X_test,y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 5)
print(X_train.shape)


# ### Selecting best Features Using SelectKBest

# In[219]:


kbest = SelectKBest(score_func=f_classif)
kbest.fit(X_train, y_train)
print("Sélection de variables :", kbest.get_support())
print("Scores de variables :", kbest.scores_)
print("Variables sélectionnées:", list(X.columns[kbest.get_support()]))
print("Variables supprimées :", list(X.columns[~kbest.get_support()]))


# In[220]:


X_train = kbest.transform(X_train)
X_test = kbest.transform(X_test)


# In[221]:


#grid_param={'n_neighbors':np.arange(1,50),'metric':['euclidean','manhattan','minkowski'],'p':np.arange(1,6)}
#grid = GridSearchCV(KNeighborsClassifier(),grid_param,cv=5)
#grid.fit(X_train,y_train)
#print(grid.score)
#print(grid.best_params_)


# In[222]:


Final_model = KNeighborsClassifier(n_neighbors=45,metric = 'manhattan', p = 1)
Final_model.fit(X_train,y_train)
Final_model.score(X_test,Y_test)


# ###### On remarque que le score est de 81% 

# In[223]:


plot_confusion_matrix(Final_model,X_test,Y_test)


# In[224]:


y_pred=Final_model.predict(X_test)
print(classification_report(Y_test,y_pred,digits=6))


# A présent on pourrait se demander si notre modèle pourrait encore avoir des meilleures performances si on lui fournissait plus de données… Pour répondre à cette question très importante il faut tracer ce qu’on appelle les courbes d’apprentissage(`learning curve`).  
# 

# ## <font color=red> Learning Curves</font>

# In[225]:


N,train_score,val_score=learning_curve(Final_model,X_train,y_train,train_sizes=np.linspace(0.1,1,10),cv=5)
print(N)


# In[226]:


plt.plot(N,train_score.mean(axis=1),label='train')
plt.plot(N,val_score.mean(axis=1),label='val')
plt.legend()
plt.show()


# On peut voir que la performance n’évolue presque plus à partir du moment où on a plus de 10000 points dans notre datasetdonc ça nous montre que le modèle produit un score entre [0.816,0.818]

# ### Logistic Regression With SelectKBest

# In[227]:


logreg=LogisticRegression(n_jobs=-1,solver='liblinear')
# entrainer ligreg
logreg.fit(X_train,y_train)


# In[228]:


logreg.score(X_test,Y_test)


# In[229]:


plot_confusion_matrix(logreg,X_test,Y_test)


# In[230]:


y_pred=logreg.predict(X_test)
print(classification_report(Y_test,y_pred,digits=6))


# ### Logistic Regression All features 

# In[231]:


categorical_features=['SEX','EDUCATION', 'MARRIAGE']
numerical_features=['LIMIT_BAL', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# In[232]:


X_train,X_test,y_train,y_test = train_test_split(X_2,Y,test_size = 0.2,random_state = 5)


# In[233]:


categorical_pipline= make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(sparse=False,handle_unknown = 'ignore'))
numerical_pipline = make_pipeline(SimpleImputer(), StandardScaler())
preprocesseur = make_column_transformer((numerical_pipline,numerical_features),(categorical_pipline,categorical_features))
model = Pipeline(steps=[("preprocessor", preprocesseur), ("classifier", LogisticRegression())])


# In[234]:


model.fit(X_train, y_train)


# In[235]:


print("model score: %.3f" % model.score(X_test, y_test))


# In[236]:


y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred,digits=6))


# In[237]:


plot_confusion_matrix(model,X_test,y_test)


# ### Logistic Regression Using only Pay_1 & Pay_2

# In[238]:


categorical_features=['SEX','EDUCATION', 'MARRIAGE']
numerical_features=['LIMIT_BAL', 'PAY_1', 'PAY_2', 'PAY_AMT1', 'PAY_AMT2','AGE', 'BILL_AMT1', 'BILL_AMT2']


# In[239]:


X_train,X_test,y_train,y_test = train_test_split(X_2,Y,test_size = 0.2,random_state = 5)


# In[240]:


categorical_pipline= make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(sparse=False,handle_unknown = 'ignore'))
numerical_pipline = make_pipeline(SimpleImputer(), StandardScaler())
preprocesseur = make_column_transformer((numerical_pipline,numerical_features),(categorical_pipline,categorical_features))
model2 = Pipeline(steps=[("preprocessor", preprocesseur), ("classifier", LogisticRegression())])


# In[241]:


model2.fit(X_train, y_train)


# In[242]:


print("model score: %.3f" % model2.score(X_test, y_test))


# In[243]:


y_pred=model2.predict(X_test)
print(classification_report(y_test,y_pred,digits=6))


# In[244]:


plot_confusion_matrix(model2,X_test,y_test)


# D'apres  la Classificaiton report on Remarque que pour la Regression logistique notre Modele est de score 0.821 ce la prouve notre Hypothese concernant la section  Data cleaning du Pay_1 & Pay_2

# ## DecisionTreeClassifier

# In[245]:


dt=DecisionTreeClassifier(random_state=0)


# In[246]:


X_train,X_test,y_train,y_test = train_test_split(X_2,Y,test_size = 0.2,random_state = 5)


# In[247]:


dt.fit(X_train,y_train)


# In[248]:


print("Score du Train",dt.score(X_train,y_train))
print("Score du test ",dt.score(X_test,y_test))


# In[249]:


grid_param={'criterion':['gini','entropie'],'max_depth':np.arange(1,10)}


# In[250]:


grid=GridSearchCV(DecisionTreeClassifier(),grid_param,cv=5)
grid.fit(X_train,y_train)


# In[251]:


print(grid.best_score_)
print(grid.best_params_)


# In[252]:


final_model=DecisionTreeClassifier(max_depth=3,criterion='gini')


# In[253]:


final_model.fit(X_train,y_train)


# In[254]:


final_model.score(X_test,y_test)


# In[255]:


plot_confusion_matrix(final_model,X_test,y_test)


# In[256]:


y_pred=final_model.predict(X_test)
print(classification_report(y_test,y_pred,digits=6))


# ## RandomForestClassifier

# In[257]:


rf_model = RandomForestClassifier(random_state = 42)

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

print("Test Accuracy: ",metrics.accuracy_score(Y_test, pred_rf))


# In[258]:


rf_confusion_matrix = metrics.confusion_matrix(Y_test, pred_rf)
sns.heatmap(rf_confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize= 12)
plt.ylabel("True Label", fontsize= 12)

plt.show()


# In[259]:


print(metrics.classification_report(Y_test, pred_rf, labels = [0, 1]))


# In[260]:


rf_pred_proba = rf_model.predict_proba(X_test)[:,1]

rf_roc_auc = metrics.roc_auc_score(Y_test, rf_pred_proba)
print('ROC_AUC: ', rf_roc_auc)

rf_fpr, rf_tpr, thresholds = metrics.roc_curve(Y_test, rf_pred_proba)

plt.plot(rf_fpr,rf_tpr, label = 'ROC_AUC = %0.3f' % rf_roc_auc)

plt.xlabel("False Positive Rate", fontsize= 12)
plt.ylabel("True Positive Rate", fontsize= 12)
plt.legend(loc="lower right")

plt.show()


# ### Voiting

# In[261]:


clf_voting = VotingClassifier(estimators=[('dt',rf_model),('lg',logreg),('knn',Final_model)])


# In[262]:


clf_voting.fit(X_train,y_train)
y_pred=clf_voting.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print("accuracy = { :2.f}  ",format(acc))


# ### Catboost

# In[263]:


cat_model = CatBoostClassifier (random_state = 42, eval_metric = 'AUC',cat_features=categorical_features)
cat_model.fit(X_train, y_train, early_stopping_rounds = 100, eval_set = [(X_test,Y_test)], cat_features = categorical_features)
pred_cat = cat_model.predict(X_test)


# In[264]:


print("Test Accuracy: ",metrics.accuracy_score(Y_test, pred_cat))


# In[265]:


cat_confusion_matrix = metrics.confusion_matrix(Y_test, pred_cat)
sns.heatmap(cat_confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize= 12)
plt.ylabel("True Label", fontsize= 12)

plt.show()


# In[266]:


print(metrics.classification_report(Y_test, pred_cat, labels = [0, 1]))


# In[267]:


cat_new_pred_proba = cat_model.predict_proba(X_test)[:,1]

cat_new_roc_auc = metrics.roc_auc_score(Y_test, cat_new_pred_proba)
print('ROC_AUC: ', cat_new_roc_auc)

cat_new_fpr, cat_new_tpr, thresholds = metrics.roc_curve(Y_test, cat_new_pred_proba)

plt.plot(cat_new_fpr,cat_new_tpr, label = 'ROC_AUC = %0.3f' % cat_new_roc_auc)
plt.xlabel("False Positive Rate", fontsize= 12)
plt.ylabel("True Positive Rate", fontsize= 12)
plt.legend(loc="lower right")

plt.show()


# In[268]:


pool = Pool(X_train, y_train, cat_features=categorical_features)

Feature_importance = pd.DataFrame({'feature_importance': cat_model.get_feature_importance(pool), 
                      'feature_names': X_train.columns}).sort_values(by=['feature_importance'], 
                                                           ascending=False)

Feature_importance


# In[269]:


plt.figure(figsize=(10,10))

sns.barplot(x=Feature_importance['feature_importance'], y=Feature_importance['feature_names'], palette = 'rocket')

plt.show()


# ### XGBOOST

# In[271]:


#gbm_param_grid = {'learning_rate': [0.01,0.1,0.5,0.9],
#'n_estimators': [200],
#'subsample': [0.3, 0.5, 0.9],"max_depth":[2,3,4,5],'colsample_bytree':[0.1,0.3,.0,8,0.9,0.15]}

gbm = xgb.XGBClassifier()
#grid_search = GridSearchCV(estimator=gbm,param_grid=gbm_param_grid,
#scoring='accuracy', cv=4, verbose=1)
#grid_search.fit(X_2,Y)
#print("Best parameters found: ",grid_search.best_params_)
#print("best accuracy found: ", np.sqrt(np.abs(grid_search.best_score_)))


# In[272]:


X_train,X_test,y_train,y_test = train_test_split(X_2,Y,test_size = 0.2,random_state = 5)


# In[273]:


reg_xgb = xgb.XGBClassifier(colsample_bytree= 0.9, learning_rate= 0.01, max_depth=3, n_estimators=200, subsample= 0.3)
reg_xgb.fit(X_train, y_train)

pred_xgb = reg_xgb.predict(X_test)
print('xgb_accuracy: {:.3f}'.format(metrics.accuracy_score(y_test, pred_xgb)))


# In[274]:


xgb.plot_importance(reg_xgb)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()


# In[275]:


xgb_confusion_matrix = metrics.confusion_matrix(y_test, pred_xgb)
sns.heatmap(xgb_confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize= 12)
plt.ylabel("True Label", fontsize= 12)

plt.show()


# ### Neural networks

# In[276]:


clf = MLPClassifier(hidden_layer_sizes=(10, ), activation='tanh', solver='sgd', 
                    alpha=0.00001, batch_size=4, learning_rate='constant', learning_rate_init=0.01, 
                    power_t=0.5, max_iter=9, shuffle=True, random_state=11, tol=0.00001, 
                    verbose=True, warm_start=False, momentum=0.8, nesterovs_momentum=True, 
                    early_stopping=False, validation_fraction=0.1, 
                    beta_1=0.9, beta_2=0.999, epsilon=1e-08)
print(clf)


# In[277]:


clf.fit(X_train,y_train)


# In[278]:


score = clf.score(X_test,y_test)


# In[279]:


print(score)


# In[280]:


y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))


# In[281]:


Grid = MLPClassifier(max_iter=100)
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
G = GridSearchCV(Grid, parameter_space, n_jobs=-1, cv=5)
G.fit(X_train,y_train) # X is train samples and y is the corresponding labels


# In[282]:


print("Best parameters found: ",G.best_params_)
print("best accuracy found: ", np.sqrt(np.abs(G.best_score_)))


# In[283]:


clf2 = MLPClassifier(hidden_layer_sizes=(20, ), activation='relu', solver='adam', 
                    alpha=0.0001,learning_rate='adaptive', learning_rate_init=0.01, 
                    verbose=True)
print(clf2)


# In[284]:


clf2.fit(X_train,y_train)


# In[285]:


score = clf2.score(X_test,y_test)


# In[286]:


print(score)


# In[310]:


pred_mlp = clf2.predict(X_test)


# In[311]:


mlp_confusion_matrix = metrics.confusion_matrix(y_test, pred_mlp)
sns.heatmap(xgb_confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize= 12)
plt.ylabel("True Label", fontsize= 12)

plt.show()


# ### Naive Bayes

# In[312]:


categorical_features=['SEX','EDUCATION', 'MARRIAGE']
numerical_features=['LIMIT_BAL', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# In[313]:


X_train,X_test,y_train,Y_test = train_test_split(X_2,Y,test_size = 0.2,random_state = 5)


# In[314]:


categorical_pipline= make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(sparse=False,handle_unknown = 'ignore'))
numerical_pipline = make_pipeline(SimpleImputer(), StandardScaler())
preprocesseur = make_column_transformer((numerical_pipline,numerical_features),(categorical_pipline,categorical_features))
model_NB = Pipeline(steps=[("preprocessor", preprocesseur), ("classifier", GaussianNB())])


# In[315]:


model_NB.fit(X_train,y_train)


# In[316]:


model_NB.score(X_test,Y_test)


# In[317]:


pred_mlp = model_NB.predict(X_test)
NB_confusion_matrix = metrics.confusion_matrix(Y_test, pred_mlp)
sns.heatmap(NB_confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize= 12)
plt.ylabel("True Label", fontsize= 12)

plt.show()


# In[318]:


y_true, y_pred = Y_test, model_NB.predict(X_test)
print(classification_report(y_true, y_pred))


# ######  Naive Bayes With SelectKBest

# In[319]:


numerical_features=['LIMIT_BAL', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']


# In[320]:


categorical_pipline= make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(sparse=False,handle_unknown = 'ignore'))
numerical_pipline = make_pipeline(SimpleImputer(), StandardScaler())
preprocesseur = make_column_transformer((numerical_pipline,numerical_features),(categorical_pipline,categorical_features))
model_NB = Pipeline(steps=[("preprocessor", preprocesseur), ("classifier", GaussianNB())])


# In[321]:


from sklearn import set_config
set_config(display='diagram')
model_NB


# In[322]:


model_NB.fit(X_train,y_train)


# In[323]:


model_NB.score(X_test,Y_test)


# In[324]:


param_grid_nb = {
    'classifier__var_smoothing': np.logspace(0,-20, num=100)
}
grid=GridSearchCV(model_NB,param_grid_nb,cv=5)
grid.fit(X_train,y_train)


# In[325]:


grid.best_score_


# In[326]:


grid.best_params_


# In[327]:


final_model_NB = grid.best_estimator_


# In[328]:


final_model_NB.fit(X_train,y_train)


# In[329]:


final_model_NB.score(X_test,Y_test)


# In[330]:


pred_mlp = model_NB.predict(X_test)
NB_confusion_matrix = metrics.confusion_matrix(Y_test, pred_mlp)
sns.heatmap(NB_confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize= 12)
plt.ylabel("True Label", fontsize= 12)

plt.show()


# In[331]:


y_true, y_pred = Y_test, final_model_NB.predict(X_test)
print(classification_report(y_true, y_pred))


# In[332]:


import pickle
pickle.dump(model, open('data.pkl', 'wb'))


# In[ ]:




