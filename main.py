import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir
from sklearn.linear_model import Ridge  
from sklearn.metrics import mean_squared_error


num_days = 30
start_date = "2023-01-01"
dates = pd.date_range(start=start_date, periods=num_days, freq='D')


np.random.seed(42)  
energy_rates = np.random.uniform(0.50, 0.20, size=num_days)


data = pd.DataFrame({
    'date': dates,
    'rate': energy_rates
})


X = data['rate'].values[:-1]  
y = data['rate'].values[1:]   


reservoir = Reservoir(units=300, sr=1.25, input_scaling=0.5)  

reservoir_states = []


for i in range(len(X)):
    state = reservoir(X[i].reshape(1, 1))  
    reservoir_states.append(state)


reservoir_out = np.concatenate(reservoir_states, axis=0)

ridge = Ridge(alpha=1e-3)  #
ridge.fit(reservoir_out, y)


predictions = ridge.predict(reservoir_out)


mse = mean_squared_error(y, predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")


plt.plot(data['date'][1:], y, label='Actual Energy Rates')
plt.plot(data['date'][1:], predictions, label='Reservoir Computing Prediction', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Energy Rate (USD/kWh)')
plt.title('Energy Rate Prediction with Reservoir Computing')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
