import csv

def estimate_price(mileage, theta0, theta1):
    """Calculates the estimated price for a given mileage."""
    return theta0 + (theta1 * mileage)

def predict():
    """Predicts the price of a car for a given mileage."""
    try:
        with open('thetas.csv', 'r') as f:
            reader = csv.reader(f)
            thetas = next(reader)
            theta0 = float(thetas[0])
            theta1 = float(thetas[1])
    except (FileNotFoundError, ValueError, IndexError):
        print("Could not load thetas. Please run train.py first.")
        theta0, theta1 = 0.0, 0.0

    while True:
        try:
            mileage = float(input("Please enter a mileage to predict the car price: "))
            if mileage < 0:
                print("Mileage cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    predicted_price = estimate_price(mileage, theta0, theta1)
    
    print(f"\nFor a mileage of {mileage:,.0f} km, the estimated price is ${predicted_price:,.2f}.")

if __name__ == "__main__":
    predict()