import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data from the Sample_Data table
df = q.cells("iris_dataset")

# Extract features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a meshgrid for visualization
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Get predictions for the meshgrid
# We can only visualize 2 features, so we'll use the first two and retrain a model on just those
model_2d = LogisticRegression()
model_2d.fit(X_train[:, :2], y_train)
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])

# Map species names to integers for plotting
species_map = {species: i for i, species in enumerate(np.unique(y))}
Z = np.array([species_map[s] for s in Z])
Z = Z.reshape(xx.shape)

# Create plot with decision boundary
fig = go.Figure()

# Add decision boundary contour
fig.add_trace(
    go.Contour(
        z=Z,
        x=np.arange(x_min, x_max, 0.01),
        y=np.arange(y_min, y_max, 0.01),
        showscale=False,
        colorscale='RdBu',
        opacity=0.4,
        contours=dict(showlines=False)
    )
)

# Add scatter points for each species
for i, species in enumerate(np.unique(y)):
    fig.add_trace(
        go.Scatter(
            x=X[y==species, 0],
            y=X[y==species, 1],
            mode='markers',
            name=species,
            marker=dict(size=10)
        )
    )


# Update layout
fig.update_layout(
    title=f'Logistic Regression Decision Boundary (Accuracy: {accuracy:.2f})',
    xaxis_title='Sepal Length',
    yaxis_title='Sepal Width',
    plot_bgcolor='white'
)

fig.show()
