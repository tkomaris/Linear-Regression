import csv

def estimate_price(mileage, theta0, theta1):
    """Calculates the estimated price for a given mileage."""
    return theta0 + (theta1 * mileage)

def train_model():
    """Trains the linear regression model using gradient descent."""
    # Hyperparameters
    learning_rate = 0.1
    iterations = 1000
    
    # Initialize thetas
    theta0 = 0.0
    theta1 = 0.0
    
    # Load data from csv file
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

    # Normalize the data for better performance
    max_km = max(row[0] for row in data)
    normalized_data = [(float(km) / max_km, price) for km, price in data]

    # Gradient Descent
    for i in range(iterations):
        sum_tmp0 = 0
        sum_tmp1 = 0
        
        for km_norm, price in normalized_data:
            error = estimate_price(km_norm, theta0, theta1) - price
            sum_tmp0 += error
            sum_tmp1 += error * km_norm
            
        # Update thetas simultaneously
        tmp0 = learning_rate * (1/m) * sum_tmp0
        tmp1 = learning_rate * (1/m) * sum_tmp1
        
        theta0 -= tmp0
        theta1 -= tmp1

    # Denormalize theta1 to work with original mileage values
    theta1 /= max_km
    
    # Save the trained parameters
    try:
        with open('thetas.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([theta0, theta1])
        print("Training complete. Thetas saved to thetas.csv")
        print(f"Final Theta0: {theta0}, Final Theta1: {theta1}")
    except IOError as e:
        print(f"Error saving thetas.csv: {e}")

if __name__ == "__main__":
    train_model()