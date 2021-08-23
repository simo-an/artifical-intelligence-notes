import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


def compute_error(w, b, points):
    total_error = 0.
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (w * x + b - y) ** 2
    return total_error / float(len(points))


def do_gradient(w, b, points, lr):
    w_grad = 0.
    b_grad = 0.
    N = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        w_grad = 2 * (w * x + b - y) * x / N
        b_grad = 2 * (w * x + b - y) / N
    new_w_grad = w - lr * w_grad
    new_b_grad = b - lr * b_grad
    return [new_w_grad, new_b_grad]


def run(w_init, b_init, points, lr, iter, error_limit):
    w_cur = w_init
    b_cur = b_init

    for i in range(iter):
        w_cur, b_cur = do_gradient(w_cur, b_cur, points, lr)
        error = compute_error(w_cur, b_cur, points)
        if error < error_limit:
            break
    return [w_cur, b_cur]


TITLE = '月广告费与月销售量数据'
XLABEL = '月广告费 万元'
YLABEL = '月销售量 万件'

dataset = np.genfromtxt('./data/data-1.csv', delimiter=',', encoding="utf-8")
x = dataset[:, 1][1:]
y = dataset[:, 2][1:]

print(x)
print(y)

f1 = plt.figure(1)
plt.title(TITLE)
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)

plt.scatter(x, y, marker='o', color='k')
plt.legend(loc='upper right')

init_w = 0
init_b = 0

data_sets = np.array([x, y]).T

print(data_sets)

init_w, init_b = run(
    w_init=init_w,
    b_init=init_b,
    points=data_sets,
    lr=0.001,
    iter=1000,
    error_limit=0.5
)

print(init_w, init_b)

r_x = np.arange(np.min(x), np.max(x))
r_y = init_w * r_x + init_b

plt.plot(r_x, r_y, color='r')

plt.show()
