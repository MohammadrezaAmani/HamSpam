import matplotlib.pyplot as plt


def create_plot(tf_idf, freq_word):
	plt.plot(list(tf_idf.keys()), list(tf_idf.values()), label="TF - IDF", color="blue")
	plt.plot(list(freq_word.keys()), list(freq_word.values()), label="Freq Word", color="red")
	plt.title("plot of accuracy")
	plt.xlabel("name of models")
	plt.ylabel("accuracy of models")
	plt.legend()
	plt.show()
	plt.savefig("accuracy.png")
