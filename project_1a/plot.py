import matplotlib.pyplot as plt

# manual plotting script

search_space = [0, 1e-12, 1e-9, 1e-6, 1e-3]
accuracy = [88.7,88.4,88.3,87.2,64.3]

plt.figure()
plt.plot(range(5), accuracy, 'bx-')
plt.gca().xaxis.set_ticks(range(5))
plt.gca().xaxis.set_ticklabels(search_space)
plt.xlabel('decay')
plt.ylabel('accuracy in %')
plt.title('decay vs accuracy')
plt.savefig('p1a_decay_accuracy.png')

plt.show()
