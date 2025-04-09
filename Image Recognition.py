import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the built-in digits dataset
digits = datasets.load_digits()

X = digits.images.reshape(len(digits.images), -1)  # Flatten 8x8 images to 64 features
y = digits.target

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to display sample images from the dataset
def show_sample_images():
    plt.figure(figsize=(8, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(digits.images[i], cmap='gray')
        plt.title(f"Label: {digits.target[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Function to show test predictions
def show_predictions():
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
        plt.title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main entry point
if __name__ == "__main__":
    show_sample_images()
    show_predictions()
