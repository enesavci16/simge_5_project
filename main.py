from urllib.request import urlopen
import requests
#from bs4 import BeautifulSoup as bts
import pandas as pd
import re
import numpy as np
import time
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

#from bokeh.io import curdoc, push_notebook, output_notebook
#from bokeh.layouts import column, layout
#from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, HoverTool
#from bokeh.plotting import figure, show
#from ipywidgets import interact, interactive, fixed, interact_manual

df = pd.read_csv("skincare_products_clean.csv")


for i in range(len(df['clean_ingreds'])):
    df['clean_ingreds'].iloc[i] = str(df['clean_ingreds'].iloc[i]).replace('[', '').replace(']', '').replace("'", '').replace('"', '')

all_ingreds = []

for i in df['clean_ingreds']:
    ingreds_list = i.split(', ')
    for j in ingreds_list:
        all_ingreds.append(j)


all_ingreds = sorted(set(all_ingreds))
#all_ingreds[0:20]

all_ingreds.remove('')
for i in range(len(all_ingreds)):
    if all_ingreds[i][-1] == ' ':
        all_ingreds[i] = all_ingreds[i][0:-1]

all_ingreds = sorted(set(all_ingreds))
#all_ingreds[0:20]

one_hot_list = [[0] * 0 for i in range(len(all_ingreds))]

for i in df['clean_ingreds']:
    k = 0
    for j in all_ingreds:
        if j in i:
            one_hot_list[k].append(1)
        else:
            one_hot_list[k].append(0)
        k += 1

ingred_matrix = pd.DataFrame(one_hot_list).transpose()
ingred_matrix.columns = [sorted(set(all_ingreds))]

#ingred_matrix

brand_list = ["111skin", "a'kin", "acorelle", "adam revolution", "aesop", "ahava", "alchimie forever",
             "algenist", "alpha-h", "ambre solaire", "ameliorate", "american crew", "anthony", "antipodes",
             "apivita", "argentum", "ark skincare", "armani", "aromatherapy associates", "aromaworks", "aromatica",
             "aurelia probiotic skincare", "aurelia skincare",
             "australian bodycare", "avant skincare", "aveda", "aveeno", "avene", "avène",
             "bakel", "balance me", "barber pro", "bareminerals", "barry m cosmetics",
             "baxter of california", "bbb london", "beautypro", "benefit", "benton", "bioderma",
             "bioeffect", "bloom & blossom", "bloom and blossom", "bobbi brown", "bondi sands", "bubble t", "bulldog", "burt's bees",
             "by terry", "carita", "caudalie", "cerave", "chantecaille", "clinique",
             "comfort zone", "connock london", "cosmetics 27", "cosrx", "cowshed", "crystal clear",
             "cult51", "darphin", "dear, klairs", "decleor", "decléor", "dermalogica", "dhc", "doctors formula",
             "dr. brandt", "dr brandt", "dr. hauschka", "dr hauschka", "dr. jackson's", "dr.jart+", "dr. lipp",
             "dr botanicals", "dr dennis", "dr. pawpaw", "ecooking", "egyptian magic",
             "eisenberg", "elemental herbology", "elemis", "elizabeth arden", "embryolisse",
             "emma hardie", "erno laszlo", "espa", "estée lauder", "estee lauder", "eucerin",
             "eve lom", "eve rebirth", "fade out", "farmacy", "filorga", "first aid beauty", "fit", "foreo",
             "frank body", "freezeframe", "gallinée", "garnier", "gatineau", "glamglow", "goldfaden md",
             "green people", "hawkins and brimble", "holika holika", "house 99", "huxley",
             "ilapothecary", "ila-spa", "indeed labs", "inika", "instant effects", "institut esthederm", "ioma", "klorane",
             "j.one", "jack black", "james read", "jason", "jo malone london", "juice beauty", "jurlique",
             "korres", "l:a bruket", "l'oréal men expert", "l'oreal men expert", "l'oréal paris", "l'oreal paris",
             "l’oréal paris", "lab series skincare for men",
             "lancaster", "lancer skincare", "lancôme", "lancome", "lanolips", "la roche-posay", "laura mercier",
             "liftlab", "little butterfly london", "lixirskin", "liz earle", "love boo",
             "löwengrip", "lowengrip", "lumene", "mac", "madara", "mádara", "magicstripes", "magnitone london",
             "mama mio", "mancave", "manuka doctor", "mauli", "mavala", "maybelline", "medik8", "men-u", "menaji", "molton brown", "moroccanoil",
             "monu", "murad", "naobay", "nars", "natio", "natura bissé", "natura bisse",
             "neal's yard remedies", "neom", "neostrata", "neutrogena", "niod", "nip+fab", "nuxe", "nyx",
             "oh k!", "omorovicza", "origins", "ortigia fico", "oskia", "ouai", "pai ", "paula's choice", "payot",
             "perricone md", "pestle & mortar", "pestle and mortar", "peter thomas roth",
             "philosophy", "pierre fabre", "pixi", "piz buin", "polaar", "prai", "project lip",
             "radical skincare", "rapideye", "rapidlash", "real chemistry", "recipe for men",
             "ren ", "renu", "revolution beauty", "revolution skincare", "rituals", "rmk", "rodial", "roger&gallet", "salcura",
             "sanctuary spa", "sanoflore", "sarah chapman", "sea magik", "sepai",
             "shaveworks", "shea moisture", "shiseido", "skin79", "skin authority", "skinceuticals",
             "skinchemists", "skindoctors", "skin doctors", "skinny tan", "sol de janeiro", "spa magik organiks",
              "st. tropez", "starskin", "strivectin", "sukin",
             "svr", "swiss clinic", "talika", "tan-luxe", "tanorganic", "tanworx", "thalgo", "the chemistry brand",
             "the hero project", "the inkey list", "the jojoba company", "the ordinary",
             "the organic pharmacy", "the ritual of namasté", "this works", "too faced", "trilogy", "triumph and disaster",
             "ultrasun", "uppercut deluxe", "urban decay", "uriage", "verso", "vichy",
             "vida glow", "vita liberata", "wahl", "weleda", "westlab", "wilma schumann", "yes to",
             "ysl", "zelens"]
brand_list = sorted(brand_list, key=len, reverse=True)





df['brand'] = df['product_name'].str.lower()
k = 0
for i in df['brand']:
    for j in brand_list:
        if j in i:
            df['brand'][k] = df['brand'][k].replace(i, j.title())
    k += 1

df['brand'] = df['brand'].replace(['Aurelia Probiotic Skincare'],'Aurelia Skincare')
df['brand'] = df['brand'].replace(['Avene'],'Avène')
df['brand'] = df['brand'].replace(['Bloom And Blossom'],'Bloom & Blossom')
df['brand'] = df['brand'].replace(['Dr Brandt'],'Dr. Brandt')
df['brand'] = df['brand'].replace(['Dr Hauschka'],'Dr. Hauschka')
df['brand'] = df['brand'].replace(["L'oreal Paris", 'L’oréal Paris'], "L'oréal Paris")


def recommender(search):
    cs_list = []
    brands = []
    output = []
    binary_list = []
    idx = df[df['product_name'] == search].index.item()
    for i in ingred_matrix.iloc[idx][1:]:
        binary_list.append(i)
    point1 = np.array(binary_list).reshape(1, -1)
    point1 = [val for sublist in point1 for val in sublist]
    prod_type = df['product_type'][df['product_name'] == search].iat[0]
    brand_search = df['brand'][df['product_name'] == search].iat[0]
    df_by_type = df[df['product_type'] == prod_type]

    for j in range(df_by_type.index[0], df_by_type.index[0] + len(df_by_type)):
        binary_list2 = []
        for k in ingred_matrix.iloc[j][1:]:
            binary_list2.append(k)
        point2 = np.array(binary_list2).reshape(1, -1)
        point2 = [val for sublist in point2 for val in sublist]
        dot_product = np.dot(point1, point2)
        norm_1 = np.linalg.norm(point1)
        norm_2 = np.linalg.norm(point2)
        cos_sim = dot_product / (norm_1 * norm_2)
        cs_list.append(cos_sim)
    df_by_type = pd.DataFrame(df_by_type)
    df_by_type['cos_sim'] = cs_list
    df_by_type = df_by_type.sort_values('cos_sim', ascending=False)
    df_by_type = df_by_type[df_by_type.product_name != search]
    l = 0
    for m in range(len(df_by_type)):
        brand = df_by_type['brand'].iloc[l]
        if len(brands) == 0:
            if brand != brand_search:
                brands.append(brand)
                output.append(df_by_type.iloc[l])
        elif brands.count(brand) < 2:
            if brand != brand_search:
                brands.append(brand)
                output.append(df_by_type.iloc[l])
        l += 1

    return (pd.DataFrame(output)[['product_name']].head(5))

recommender('JASON Soothing Aloe Vera Body Wash 887ml')

import streamlit as st
from PIL import Image


image_ist=Image.open('akademi.jpeg')

image_2 = Image.open('baslik_1.jpeg')
image_3 = Image.open('baslik_2.jpeg')
st.image(image_ist)
st.image(image_2)
st.image(image_3)
st.write('The recommendation engine enables users to make better decisions on which product to purchase, as many recommendations contain products that are better value for money. It also has the potential to improve business for less popular brands by recommending their products.')

a = st.text_input('Please enry product')

#recommender("JASON Soothing Aloe Vera Body Wash 887ml")

predict=st.button("Predict")
if predict:
    sonuc=recommender(a)
    print(sonuc)
    st.write(sonuc)

