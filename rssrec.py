import feedparser
import math
import nltk
import numpy
import operator
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import *
from decimal import *

urluserprofile = raw_input("Enter the url of the RSS feed for user profile creation: ")

#"http://halley.exp.sis.pitt.edu/comet/utils/_rss.jsp?month=1&year=2014&user_id=1"

urltargetset = raw_input("Enter url of the target RSS feed: ")

#"http://halley.exp.sis.pitt.edu/comet/utils/_rss.jsp?month=10&year=2013"

#list of word-frequencies pairs for each document
terms_in_docs = {}

# term frequency across all documents
term_count = {}

# number of documents a term is in
doc_freq = {}

# user profile term weights
userprofile = {}
totalweights = 0

feeduserprofile = feedparser.parse(urluserprofile)
feedtargetset = feedparser.parse(urltargetset)

#title = feed.entries[i].title
#pubdate = feed.entries[i].published
#link = feed.entries[i].link

#tokenize, remove stop words, stem document
def tokenize (content):
	raw = nltk.clean_html(content)
	tokens = RegexpTokenizer(r'\w+').tokenize(raw.lower())
	words_remaining = []
	for token in tokens:
		if token not in stopwords.words('english'):
			words_remaining.append(token)
	
	stemmer = PorterStemmer()
	stemmed_words = []
	
	for word in words_remaining:
		stemmed_words.append(stemmer.stem(word))
		
	return stemmed_words

#determine term frequency of individual document
def term_freq_doc(content):
	freq_dist = nltk.FreqDist()
	for word in content:
		freq_dist.inc(word)
		if word not in term_count:
			term_count[word] = 1
		else:
			term_count[word] += 1
	for word in freq_dist:
		if word not in doc_freq:
			doc_freq[word] = 1
		else:
			doc_freq[word] += 1
	
	return freq_dist

#determine term frequency for all documents in rss feed
def term_freq_all(feed):
	for i in range(0, len(feed['entries'])):
		content = feed.entries[i].description
		stemmed_words = tokenize(content)
		terms_in_docs[i] = term_freq_doc(stemmed_words)	
		
term_freq_all(feeduserprofile)

#calculate tf*idf for each term in document
def tfidf(i, feed):
	global totalweights
	N = Decimal(len(feed['entries']))
	for term in terms_in_docs[i]:
		idf = math.log10((N/doc_freq[term]))
		tfidf = idf*terms_in_docs[i][term]
		terms_in_docs[i][term] = tfidf
		if term not in userprofile:
			userprofile[term] = tfidf
			totalweights += tfidf
		else:
			userprofile[term] += tfidf
			totalweights += tfidf

#calculate tf*idf for all documents
def tfidf_all(feed):
	for i in range(0,len(feed['entries'])):
		tfidf(i, feed)	
		
tfidf_all(feeduserprofile)


terms_in_docs = {}
sims_docs = {}

#create word-frequency for target set
term_freq_all(feedtargetset)

#calculate cosine of similarity for one document		
def calc_cos_sim(doc):
	vector_user = []
	vector_target = []
	
	for term in terms_in_docs[doc]:
		vector_target.append(terms_in_docs[doc][term])
		if term in userprofile:
			vector_user.append(userprofile[term])		
		else:
			vector_user.append(0)
	return float(dot(vector_user,vector_target)/(norm(vector_user)*norm(vector_target)))

#calculate cosine of similarity for all documents in target set
def all_sims():	
	global sims_docs
	for i in range(0, len(feedtargetset['entries'])):
		sims_docs[i] = calc_cos_sim(i)
	
all_sims()

#sort documents from lowest cosine of similarity to highest
sorted_sims_docs = sorted(sims_docs.iteritems(), key=operator.itemgetter(1))

#print cosine of similarity and article title from highest to lowest
def print_titles_ordered(sorted):
	max = len(sorted_sims_docs) - 1
	count = max
	while (count >=0):
		print "Similarity is " + str(sorted[count][1])
		print feedtargetset.entries[sorted[count][0]].title
		count -= 1

print_titles_ordered(sorted_sims_docs)



	


	




