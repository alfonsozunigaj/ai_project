import matplotlib.pyplot as plt

def time_graph(x, heights):
    plt.bar(x, heights)
    plt.xlabel('Capas Ocultas (cant.)')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Demora Según Cantidad de Capas Ocultas')
    plt.show()

x = [1, 2, 3, 4]
y = [390, 500, 350, 444]

time_graph(x, y)