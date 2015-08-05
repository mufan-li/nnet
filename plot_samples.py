# plot_samples

for i in range(10):
	index = rd.randint(0,10000-1)
	temp = predict([index])
	tempclass = temp[1]
	plt.figure()
	plt.imshow(np.reshape(valid_set_x.get_value()[index],(28,28)))
	plt.title('Validation Case: ' + str(index) + ' - Prediction: ' +
				str(tempclass) )

plt.show()
