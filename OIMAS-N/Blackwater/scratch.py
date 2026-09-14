import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fn = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Blackwater/callibration_output/DBSP4_call_df.csv"

df = pd.read_csv(fn, index_col=0)
df = df.sort_values(by='RMSE_C', ascending=True)

# normalize errors
lhc_rmse_C_n        = df['RMSE_C'] / np.std(df['RMSE_C'])
lhc_rmse_Z_n        = df['RMSE_Z'] / np.std(df['RMSE_Z'])

logL = -0.5 * (
        1.0 * lhc_rmse_C_n ** 2 +
        1.0 * lhc_rmse_Z_n ** 2
 )

L = np.exp(logL)

sc = plt.scatter(df['RMSE_C'], df['RMSE_Z'], 5, L)

#for i in df.index:
#    plt.text(df['RMSE_C'].loc[i], df['RMSE_Z'].loc[i], i)

plt.colorbar(sc)
plt.xlabel('RMSE_C')
plt.ylabel('RMSE_Z')
plt.show()