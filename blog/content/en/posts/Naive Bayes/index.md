---
title: "Naive Bayes for Text Classification"
date: 2024-09-14
author: "Paul Kroeger"
description: "Using Naive Bayes for classifying  20 Newsgroups dataset."
toc: true
tags:
---

## Introduction
## Introduction

Naive Bayes classifiers are simple yet powerful supervised learning algorithms commonly used for text classification tasks such as sentiment analysis, spam detection, language identification, and more. In this blog post, we will explore the theory behind Naive Bayes classifiers and implement one to categorize text in the [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/). When first learning about Naive Bayes, I found the book [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) by Daniel Jurafsky and James H. Martin extremely helpful.

## Why Naive?

Naive Bayes classifiers are probabilistic models that compute a probability distribution over possible classes for a given input document. Their beauty lies in their simplicity. However, natural language is notoriously complex, so to keep the approach manageable, we need to make some simplifying assumptions. Probably the most drastic simplification is in how input texts are represented. Naive Bayes models represent a text as a **bag of words**—that is, they consider each word (or token) in the input as part of an unordered collection, disregarding grammar and word order. Furthermore, unlike mathematical sets, this "bag" allows for multiple occurrences of the same word.

For example, if we use spaces to separate words (we will later ignore punctuation), the sentence “The Answer to the Great Question... Of Life, the Universe and Everything... Is... Forty-two” becomes the case-insensitive bag of words:

```
"answer", "forty-two", "the", "great", "of", "question", "to", "the", "life", "the", "universe", "is", "everything", "and"
```

Note that the order of words is completely arbitrary, and words may appear multiple times.

This modeling choice has a significant implication: our model will not be able to incorporate sentence structure or the positions of words into its predictions. It relies solely on word frequencies.

## Why Bayes?

Naive Bayes classifiers are probabilistic models that compute a probability distribution over possible classes for a given input document. Let $\mathcal{C}$ be the set of classes, and let $d$ be the document we want to classify. Our goal is to find the class $c \in \mathcal{C}$ that maximizes the posterior probability $P(c \mid d)$. Formally, we define:

$$
    \hat{c} = \arg\max_{c \in \mathcal{C}} P(c \mid d)
$$

To estimate $P(c \mid d) we apply **Bayes' theorem**. According to Bayes' theorem:

$$
    P(c \mid d) = \frac{P(d \mid c) P(c)}{P(d)}
$$

Substituting this into our classification rule, we get:

$$
    \hat{c} = \arg\max_{c \in \mathcal{C}} \frac{P(d \mid c) P(c)}{P(d)} = \arg\max_{c \in \mathcal{C}} P(d \mid c) P(c)
$$

The second equality holds because $P(d)$ is constant with respect to $c$ and does not affect the maximization. Here, $P(d \mid c)$ is called **likelihood**, and $P(c)$ is refered to as **prior** probability of class $c$.

### Estimating the Prior

The prior $P(c)$ can be estimated using the relative frequency of class $c$ in the training set:

$$
    \hat{P}(c) = \frac{\text{Number of documents in class } c}{\text{Total number of documents}}
$$

### Estimating the Likelihood

Estimating $P(d \mid c)$ directly is challenging due to the vast number of possible documents. To simplify, we make the **Naive Bayes assumption**: the features (words) are conditionally independent given the class. Representing our document $d$ as a sequence of words $w_1, w_2, \ldots, w_n$, this assumption gives the second equality below:

$$
    P(d \mid c) = P(w_1, w_2, \ldots, w_n \mid c) = \prod_{w \in d} P(w \mid c)
$$

This assumption is strong and not generally true, as words in a document can be dependent (e.g., "not" and "hungry" in "I am not hungry"). However, since we're using a bag-of-words representation that ignores word order and syntax anyways, this simplification will turn out to be acceptable and it makes computation feasible.

Each term $P(w_i \mid c)$ answers the question: *"How likely is word $w_i$ to appear in a document of class $c$?"* We can estimate this probability by calculating the relative frequency of $w_i$ in all documents of class $c$:

$$
    \hat{P}(w_i \mid c) = \frac{\text{Count}(w_i, c)}{\sum_{w \in V} \text{Count}(w, c)}
$$

Here, $V$ denotes the **vocabulary**, the set of all distinct words in the training documents.

### Putting Everything Together

Combining our estimates of the prior and likelihood, we arrive at:

$$
    \hat{c} = \arg\max_{c \in \mathcal{C}} \hat{P}(c) \prod_{w \in d} \hat{P}(w \mid c)
$$

## Final Details

There are some important details we need to address.

Looking at our last equation for $\hat{c}$, we have a potentially large product, which is inefficient to compute. Moreover, since the factors of the product are probabilities less than one, multiplying many of them can lead to numerical underflow. To mitigate this, we employ a common technique: we take the logarithm of the expression. The logarithm is a strictly increasing function, so it does not change the location of the maximum. This transformation allows us to convert the product into a sum, which is more computationally efficient and numerically stable. Therefore, we can rewrite our equation as:

$$
    \hat{c} = \arg\max_{c \in \mathcal{C}} \hat{P}(c) \prod_{w \in d} \hat{P}(w \mid c) = \arg\max_{c \in \mathcal{C}} \left( \log \hat{P}(c) + \sum_{w \in d} \log \hat{P}(w \mid c) \right)
$$

However, taking the logarithm introduces another problem. Specifically, the estimated likelihood $\hat{P}(w \mid c) = \frac{\text{count}(w, c)}{\sum_{w' \in V} \text{count}(w', c)}$ might be zero if $\text{count}(w, c)$ is zero (i.e., the word $w$ has not appeared in the training data for class $c$). Since the logarithm of zero is undefined, we need to ensure that our estimated likelihoods are never zero. Intuitively, while the absence of a word can make a class less likely, it should not make the probability of that class zero outright. To avoid this issue, we apply a technique called **smoothing**.

There are several smoothing techniques available, but for simplicity, we'll use **Laplace smoothing** (also known as add-one smoothing). We adjust our estimate of the likelihood as follows:

$$
    \hat{P}(w \mid c) = \frac{\text{count}(w, c) + 1}{\sum_{w' \in V} \left( \text{count}(w', c) + 1 \right)} = \frac{\text{count}(w, c) + 1}{\left( \sum_{w' \in V} \text{count}(w', c) \right) + |V|}
$$

Here, $|V|$ is the size of the vocabulary, i.e., the number of unique words in our training data. By adding 1 to each count in the numerator and adding $|V|$ to the denominator, we ensure that no probability estimate is zero.

Finally, we need to handle words in the test documents that are not present in our vocabulary (i.e., words we haven't seen during training). Since we cannot assign probabilities to these unseen words, our solution is to ignore them during classification. That is, we only consider words that are in our vocabulary when computing the sum of log-likelihoods.

This approach ensures that all terms in our calculation are well-defined and that our model can robustly handle new data.

## References
[Speech and Language Processing, Daniel Jurafsky, James H. Martin](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf), Chapter 4