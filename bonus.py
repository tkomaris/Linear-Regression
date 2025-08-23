import csv
import matplotlib.pyplot as plt

def estimate_price(mileage, theta0, theta1):
    """Calculates the estimated price for a given mileage."""
    return theta0 + (theta1 * mileage)

def calculate_r_squared(data, theta0, theta1):
    """Calculates the R-squared value to measure precision."""
    m = len(data)
    if m == 0:
        return 0.0
    
    mean_price = sum(price for _, price in data) / m
    
    ss_total = sum((price - mean_price) ** 2 for _, price in data)
    ss_residual = sum((price - estimate_price(km, theta0, theta1)) ** 2 for km, price in data)
    
    if ss_total == 0:
        return 1.0
        
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def bonus_features():
    """Trains, visualizes, and calculates the precision of the model."""
    learning_rate = 0.1
    iterations = 1000
    
    theta0 = 0.0
    theta1 = 0.0
    
    try:
        with open('data.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            data = [(int(row[0]), int(row[1])) for row in reader]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data.csv: {e}")
        return

    m = len(data)
    if m == 0:
        print("data.csv is empty or invalid.")
        return
        
    mileages = [row[0] for row in data]
    prices = [row[1] for row in data]

    max_km = max(mileages)
    normalized_data = [(float(km) / max_km, price) for km, price in data]

    for i in range(iterations):
        sum_tmp0 = 0
        sum_tmp1 = 0
        
        for km_norm, price in normalized_data:
            error = estimate_price(km_norm, theta0, theta1) - price
            sum_tmp0 += error
            sum_tmp1 += error * km_norm
            
        theta0 -= learning_rate * (1/m) * sum_tmp0
        theta1 -= learning_rate * (1/m) * sum_tmp1

    theta1 /= max_km
    
    print(f"Training complete.")
    print(f"Final Theta0: {theta0}, Final Theta1: {theta1}")

    precision = calculate_r_squared(data, theta0, theta1)
    print(f"Model Precision (R-squared): {precision:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(mileages, prices, color='blue', label='Actual Car Prices')
    
    regression_line_x = range(0, max(mileages) + 50000, 1000)
    regression_line_y = [estimate_price(x, theta0, theta1) for x in regression_line_x]
    
    plt.plot(regression_line_x, regression_line_y, color='red', linewidth=2, label='Linear Regression Line')
    plt.title('Car Price vs. Mileage')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    bonus_features()