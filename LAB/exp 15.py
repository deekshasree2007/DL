#An email filtering system identifies 30% of incoming emails as spam. The word "prize" appears in 70% of spam emails and in 10% of non-spam emails.
import matplotlib.pyplot as plt
 
# Step 1: Given probabilities
p_spam = 0.30                    # P(Spam)  prior
p_word_given_spam = 0.70         # P(prize | Spam)
p_word_given_not_spam = 0.10     # P(prize | Not Spam)
 
# Step 2: Probability of non-spam
p_not_spam = 1 - p_spam
 
# Step 3: Total probability that an email contains the word "prize"
p_word = p_word_given_spam * p_spam + p_word_given_not_spam * p_not_spam
 
# Step 4: Bayes' theorem  P(Spam | prize)
posterior = (p_word_given_spam * p_spam) / p_word
 
print("Prior probability P(Spam)             =", p_spam)
print("Probability of word 'prize' P(prize)  =", round(p_word, 4))
print("Posterior probability P(Spam | prize) =", round(posterior, 4))
print("Posterior in percentage               =", round(posterior * 100, 2), "%")
 
# Step 5: Visualize prior vs posterior
labels = ["Prior P(Spam)", "Posterior P(Spam | prize)"]
values = [p_spam, posterior]
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, values, color=["steelblue", "tomato"], edgecolor="black")
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center")
plt.ylabel("Probability")
plt.title("Probability of Spam: Before and After Seeing the Word 'prize'")
plt.ylim(0, 1)
plt.show()
