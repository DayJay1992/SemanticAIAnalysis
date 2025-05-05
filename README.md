This project is an attempt to highlight diffenrences between German AI generated academic texts and human academic texts in the field of Germanic linguistics by semantic means. In a nutshell, the project consists of two main scripts (NLTK/scripts/main.py) and (NLTK/scripts/clustering.py), both with different focuses. There are additional scripts that produce more convenient overviews over the data.


**The corpus**
Consists of human texts and AI generated texts from different Models and with two different prompts, namely:

| Service           |Models    | Prompts |
| :---------------- | :------: | ----: |
| OpenAI            |   ChatGPT4o and ChatGPTo1   | A and B |
| Google            |   Gemini Flash 2.0 and Flash 2.0 Thinking   | A and B |
| Deepseek          |  Deepseek V3 and R1   | A and B |

Each human text consists of the introduction of a peer reviewed academic paper out of the field of germanic linguistics. All human texts can loosely be categorized as syntactic papers that deal with the left periphery of the sentence (V2, V3, pre-prefield, etc.), thus they are relatively similar but not topically identical. The AI models have been prompted to generate introductions to the exact same topics. Two separate prompts have been used, producing two independent texts for each human texts. Prompt A was a more simplistic prompt in a style like "Generate an academic introduction in the field of linguistics about ((topic))". Prompt B was more specific, asking specifically for academic tone, harvard quotation style and academic structure for the introduction text. For each human text, a total of 12 AI texts have been generated about exactly the same topic (6 Models á 2 prompts). 25 human texts have been extracted, resulting in a total of a combined 325 texts as a corpus. The corpus can be found in /NLTK/scripts/corpus/texte.json


**main.py**
basically reads, filters, tokenizes and lemmatizes all texts and counts the total occurences of all relevant words and n-grams and calculates some stilometric features. The result is saved in textanalyse_gesamt.xlsx
The scripts does several additional runs where it does the same but calculates adjectives (ADJ), adverbs (ADV), Nouns (NOUN) and Verbs (VERBS) seperately for convencience. The excel files only contain the top 100 lemmas of all models for performance reasons. The complete analysis is saved in /NLTK/scripts/unique_lemmata_output/ as a.txt-file for each model and POS. The used language model is *de_core_news_lg* from the *spacy* package.

**clustering.py**
clustering.py tries to put all lemmas into categories of lemmas with similar semantic meaning. "Semantic meaning", in this case, is the embedding vector assigned to each lemma by the *de_core_news_lg* model. In this case, if two lemmas have a cosine similarity of at least 0.7, they are put into the same semantic category. Then, the occurences of each category in every text sort is counted. The results and the global categories are printed in the NLTK/scripts/Kategorisierungen_*-Folders. Again, one run takes all POS into accounts (NLTK/scripts/Kategorisierungen_Alle), but there are additional runs for each POS (and different combinations of POS, such as adjectives and adverbs) seperately. 

**abweichungen_kategorien.py**
based on the files produced by clustering.py, this script creates a plot that displays the top 30 over- and underrepresented lemmas compared to avarage appearance. The plot can be found in the resepctive NLTK/scripts/Kategorisierungen_*-Folders as abweichungen_kategorien_plot.png. It creates plots for all NLTK/scripts/Kategorisierungen_*-Folders automatically

**Heatmap_Kategorien.py**
based on the files produced by clustering.py, this script creates an interactice heatmap that displays the top 100 occurences of each semantic category in each model. The heatmaps can be found in the resepctive NLTK/scripts/Kategorisierungen_*-Folders as interaktive_heatmap.html. It creates heatmaps for all NLTK/scripts/Kategorisierungen_*-Folders automatically.

##Acknowledgments

**-Yannic Pixberg - for his contribution of texts to the corpus database**





