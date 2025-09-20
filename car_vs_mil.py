import numpy as np

# Feature: mileage (in 1000 km)
x = np.array([10, 20, 30, 40, 50], dtype=float)

# Target: price (in $1000s)
y = np.array([90, 80, 70, 60, 50], dtype=float)

# initial values
w = 0.0
b = 0.0

# linear function
def linear_function(x, w, b):
    return np.dot(w, x) + b

# cost function
def compute_cost(x, y, w, b): 
    m = x.shape[0] 
    f = linear_function(x, w, b)  # predicted values
    cost = (f - y) ** 2           # squared errors
    total_cost = (1 / (2 * m)) * np.sum(cost)  # average over m
    return total_cost

# gradient function
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0.0
    dj_db = 0.0
    for i in range(m):
        f = linear_function(x[i], w, b)  # use one example at a time
        dj_dw += (f - y[i]) * x[i]
        dj_db += (f - y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

# gradient descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}, w {w:.4f}, b {b:.4f}")
    return w, b

# train model
w_final, b_final = gradient_descent(x, y, w, b, alpha=0.001, num_iters=1000)
print(f"\nFinal parameters: w = {w_final:.4f}, b = {b_final:.4f}")

# prediction
mileage = 35  # in 1000 km
predicted_price = linear_function(mileage, w_final, b_final)
print(f"Predicted price for a car with {mileage*1000} km: ${predicted_price*1000:.2f}")
