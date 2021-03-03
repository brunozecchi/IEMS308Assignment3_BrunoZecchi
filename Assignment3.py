#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:25:05 2021

@author: Bruno Zecchi
"""

#%% Open packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import sys
sys.stdout = open("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/out_Assign3.txt", "w")

# %% Load Data
dfCEO = pd.read_csv("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/labels/ceo.csv",encoding="latin-1",names=["First","Last"])
dfComp = pd.read_csv("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/labels/companies.csv",names=["Name"])
dfPerc = pd.read_csv("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/labels/percentage.csv",encoding="latin-1",names=["Percentage"])

#%% Data Exploration
print(dfCEO.head(10))
print(dfComp.head(10))
print(dfPerc.head(10))

print(dfCEO.isnull().any())
print(dfComp.isnull().any())
print(dfPerc.isnull().any())

#%% Fix CEO table
dfCEO.loc[pd.isnull(dfCEO["Last"]),"Last"] = ""
dfCEO.loc[pd.isnull(dfCEO["First"]),"First"] = ""

#Remove spaces from end of first names
for x in range(len(dfCEO)): 
    if dfCEO["First"][x]  != "" and dfCEO["First"][x][-1]==" ":
        dfCEO["First"][x] = dfCEO["First"][x][:-1]



#%%
dfCEO["Name"] = dfCEO["First"] + " " + dfCEO["Last"]
dfPerc["Name"] = dfPerc["Percentage"]



# Remove again spaces at end and beginning from Name
for x in range(len(dfCEO)): 
    if dfCEO["Name"][x][-1]==" ":
        dfCEO["Name"][x] = dfCEO["Name"][x][:-1]
        
    if dfCEO["Name"][x][0]==" ":
        dfCEO["Name"][x] = dfCEO["Name"][x][1:]
     
dfCEO = dfCEO[dfCEO["Name"]!=""]
dfCEO.drop_duplicates(subset=["Name"],inplace=True)


#%% Build lists of ceo names, used later for comparison
ceo_first = []
ceo_last = []
for name in dfCEO["Name"]:
    u = re.findall(" ", name)
    if len(u)==1:
        ind = name.index(" ")
        ceo_first.append(name[:ind])
        ceo_last.append(name[ind+1:])
    if len(u)==0:
        ceo_first.append(name)
        ceo_last.append(name)
    if len(u)>1:
        ind = name.index(" ")
        ceo_first.append(name[:ind])
        ind2 = name.rfind(" ")
        ceo_last.append(name[ind2+1:])
        
ceo_full = dfCEO["Name"]
#%%Clean companies dataset
dfComp.drop_duplicates(subset="Name",inplace=True)



#%% Clean percentage dataset
dfPerc.drop_duplicates(subset="Name",inplace=True)

#%% Combine dataframes
dfCEO['Type'] = 1
dfComp['Type'] = 2
dfPerc['Type'] = 3

    
df = pd.concat([dfCEO[["Name","Type"]],dfPerc[["Name","Type"]],dfComp[["Name","Type"]]],ignore_index=True)
df = df.sample(frac=1)




#%% Collect negative samples
famous = pd.read_csv("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/famous.csv",encoding="latin-1",names=["Name"])
famous = famous[~famous["Name"].isin(ceo_full)] #Remove CEO famous names
famous = famous.sample(frac=1) #shuffle up names

companyNeg = pd.read_csv("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/company_fict.csv",encoding="latin-1",names=["Name"])
companyNeg.drop(0,inplace=True)
percNeg = pd.read_csv("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/percent_neg.csv",encoding="latin-1",names=["Name"])

locations = pd.read_csv("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/uscities.csv",encoding="latin-1")
locations = locations[["city","state_name"]][0:713]
locs= pd.concat([locations.city,locations.state_name],ignore_index=True)



wordsRandom = pd.read_csv("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/wordlist.csv",encoding="latin-1",names=["Name"])
wordsRandom["Type"] = 4

dfNeg = pd.concat([famous.Name,companyNeg.Name,percNeg.Name,locs],ignore_index=True)
dfNeg = dfNeg.sample(frac=1)

dfNeg = dfNeg.to_frame()
dfNeg["Type"] = 4
dfNeg.reset_index(drop=True,inplace=True)
dfNeg.rename(columns={0:"Name"},inplace=True)

#%%
dffinal = pd.DataFrame()

for x in range(len(df)):
    tipo = df.iloc[x].Type
    name = df.iloc[x].Name
    
    nameTok = word_tokenize(name)
    for n in nameTok:
        insert = df.iloc[x]
        insert.Name = n
        dffinal = dffinal.append(insert)
    
    dffinal = dffinal.append(wordsRandom.sample(5)) #spread out data
    
    name2 = dfNeg.iloc[x].Name
    nameTok2 = word_tokenize(name2)

    for n2 in nameTok2:
        insert2 = dfNeg.iloc[x]
        insert2.Name = n2
        dffinal = dffinal.append(insert2)
 
    dffinal = dffinal.append(wordsRandom.sample(5)) #spread out data

dffinal.reset_index(drop=True,inplace=True)
dffinal.Name = dffinal.Name.astype(str)
#%%

#Var1: Is in ceo Names

compnames = dfComp.Name
#Var3: Is in compnames


#Percentages
percentages = dfPerc.Name
#Var2: Is in percentages


compEnds = ["Co","Co.","International","Inc","Inc.","Ltd","Ltd.","LLC","LLC."]

def dfbuilder(dfFinal):
    dfFinal["var1"] = 0
    dfFinal["var2"] = 0
    dfFinal["var3"] = 0
    dfFinal["var4"] = 0 #is next token % or "percent" ?
    dfFinal["var5"] = 0 #is token capitalized?
    dfFinal["var6"] = 0 #is token first letter capital + next token first letter capitalized?
    dfFinal["var7"] = 0 #next tokens contain Co.,Inc,Ltd,etc
    
    for x in range(len(dfFinal)-5):
       
        name = dfFinal.Name[x]
        name2 = dfFinal.Name[x] + " " + dfFinal.Name[x+1]
        name3 = dfFinal.Name[x] + " " + dfFinal.Name[x+1] + " " + dfFinal.Name[x+2]
        name4 = dfFinal.Name[x] + " " + dfFinal.Name[x+1] + " " + dfFinal.Name[x+2]+ " " + dfFinal.Name[x+3]
        
        #CEO Names ---------------
        if (name in ceo_last):
            dfFinal["var1"][x]=1
    
        if(ceo_full==name2).any():
    
            dfFinal["var1"][x]=1
            dfFinal["var1"][x+1]=1
            
      
        if(ceo_full==name3).any():
            dfFinal["var1"][x]=1
            dfFinal["var1"][x+1]=1
            dfFinal["var1"][x+2]=1
            
    
        if(ceo_full==name3).any():
            dfFinal["var1"][x]=1
            dfFinal["var1"][x+1]=1
            dfFinal["var1"][x+2]=1
            dfFinal["var1"][x+3]=1
        
        
        
        #Percentages --------------
        if(percentages==(name+dfFinal.Name[x+1])).any():
            dfFinal["var2"][x]=1
            dfFinal["var2"][x+1]=1
    
        
        if dfFinal.Name[x+1]  in ["%","percent","Percent"]:
            dfFinal["var4"][x]=1
            dfFinal["var4"][x+1]=1
        
        
        
        #Company Names -------------
        if (compnames==name).any():
            dfFinal["var3"][x]=1
           
        if(compnames==name2).any():
            dfFinal["var3"][x]=1
            dfFinal["var3"][x+1]=1
           
        if(compnames==name3).any():
            dfFinal["var3"][x]=1
            dfFinal["var3"][x+1]=1
            dfFinal["var3"][x+2]=1
           
        if(compnames==name3).any():
            dfFinal["var3"][x]=1
            dfFinal["var3"][x+1]=1
            dfFinal["var3"][x+2]=1
            dfFinal["var3"][x+3]=1
          
        if (dfFinal.Name[x+1] in compEnds):
            dfFinal['var7'][x] = 1
            dfFinal['var7'][x+1] = 1
            
        if (dfFinal.Name[x+2] in compEnds):
            dfFinal['var7'][x] = 1
            dfFinal['var7'][x+1] = 1
            dfFinal['var7'][x+2] = 1
        
        if (dfFinal.Name[x+3] in compEnds):
            dfFinal['var7'][x] = 1
            dfFinal['var7'][x+1] = 1
            dfFinal['var7'][x+2] = 1 
            dfFinal['var7'][x+3] = 1 
            
        #Other ---------------------
        if name[0].isupper():
            dfFinal['var5'][x] = 1
        
        if name[0].isupper() and dfFinal.Name[x+1][0].isupper():
            dfFinal['var6'][x] = 1
            
    return dfFinal
      
dffinal = dfbuilder(dffinal)      

    
#%% Build Training Model
train = dffinal.iloc[0:70000]
test = dffinal.iloc[70001:]
  
xtrain = train.iloc[:,2:]
ytrain = train.iloc[:,1]    

xtest = test.iloc[:,2:]
ytest = test.iloc[:,1]
  
from sklearn.linear_model import LogisticRegression  

classifier = LogisticRegression().fit(xtrain,ytrain)
ypred = classifier.predict(xtest)

print("Correct Predictions:")
print(sum(ypred==ytest) / len(ytest))

print("Correct CEO Names Predictions")
print(sum(ypred[ytest==1] == ytest[ytest==1]) / len(ytest[ytest==1]) )
        
print("Correct Company Names Predictions")
print(sum(ypred[ytest==2] == ytest[ytest==2]) / len(ytest[ytest==2]) )

print("Correct Percentages Predictions")
print(sum(ypred[ytest==3] == ytest[ytest==3]) / len(ytest[ytest==3]) )        

print("Correct 'Other' Predictions")
print(sum(ypred[ytest==1] == ytest[ytest==1]) / len(ytest[ytest==1]) )

print("Incorrect Predictions")
print(sum(ypred != ytest) / len(ytest) )

  
#%% Lastly, apply model to text files
import os
filenames = []
filenames.append(os.listdir("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/BI-articles/2013/") )
filenames.append(os.listdir("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/BI-articles/2014/") )
year = ["2013","2014"]
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))

fileCEOs = []
fileComps = []
filePercs = []

finalCEO = pd.Series()
finalCompany = pd.Series()
finalPercentage = pd.Series()
for yr in [0,1]:
        
    for filname in filenames[yr]:
        filpath = "/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/BI-articles/" + year[yr] + "/"+filname
        print(filpath)
        text = open(filpath,"r",encoding='latin-1').read()
        text = text.replace("\r"," ").replace("\n"," ").replace("  "," ")
    
        tokens = word_tokenize(text)
        result = [i for i in tokens if not i in stop_words]
        
        dfText = pd.DataFrame()
        dfText["Name"]=result
        
        dfText2 = dfbuilder(dfText)
        extractArray = classifier.predict(dfText2.iloc[:,1:])
        extract = dfText2.Name.to_frame()
        extract["Type"] = extractArray
        
        extract.reset_index(drop=True,inplace=True)
        if extract.Type[0]==1:
                fileCEOs.append(extract.Name[0])
        elif extract.Type[0]==2:
            fileComps.append(extract.Name[0])
        elif extract.Type[0]==3:
            filePercs.append(extract.Name[0])
                
        for x in range(1,len(extract)):
            if extract.Type[x]==extract.Type[x-1]:
                if extract.Type[x]==1:
                    fileCEOs[-1]=fileCEOs[-1]+" "+extract.Name[x]
                elif extract.Type[x]==2:
                    fileComps[-1]=fileComps[-1]+" "+extract.Name[x]
                elif extract.Type[x]==3:
                    filePercs[-1]=filePercs[-1]+ extract.Name[x]
            else:
                if extract.Type[x]==1:
                    fileCEOs.append(extract.Name[x])
                elif extract.Type[x]==2:
                    fileComps.append(extract.Name[x])
                elif extract.Type[x]==3:
                    filePercs.append(extract.Name[x])
        
        finalCEO = finalCEO.append(pd.Series(fileCEOs)).drop_duplicates()
        finalCompany = finalCompany.append(pd.Series(fileComps)).drop_duplicates()
        finalPercentage = finalPercentage.append(pd.Series(filePercs)).drop_duplicates()
                
            
        
finalCEO.to_csv('/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/finalCEO.txt', sep=' ', index=False)
finalCompany.to_csv('/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/finalCompany.txt', sep=' ', index=False)
finalPercentage.to_csv('/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/finalPercentage.txt', sep=' ', index=False)


