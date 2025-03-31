# Configuration parameters for the pair trading system

# Data parameters
DATA_PATH = 'data/stock_data.csv'
TRAIN_YEARS = 3
TOTAL_YEARS = 5

# Pair generation parameters
COINTEGRATION_PVALUE = 0.05
USE_PARALLEL = True
PARALLEL_JOBS = -1  # -1 uses all available cores

# Trading parameters
ZSCORE_WINDOW = 20
ZSCORE_THRESHOLD = 1.5
POSITION_SIZE = 1.0

# Kalman filter parameters
KALMAN_DELTA = 1e-5
KALMAN_R = 0.001

# Filter parameters
# SSD Filter
SSD_TOLERANCE = 0.5

# Fractional Cointegration Filter
FRAC_D_MIN = 0.2
FRAC_D_MAX = 0.8
FRAC_ADF_THRESHOLD = -3.5

# Half-Life Filter
HALF_LIFE_MIN = 1
HALF_LIFE_MAX = 20

# Hurst Exponent Filter
HURST_MIN = 0.0
HURST_MAX = 0.49

# Volatility Regime Filter
MAX_HIGH_VOL_PCT = 0.65
VOL_N_REGIMES = 3
VOL_WINDOW = 126

# Portfolio optimization parameters
OPTIMIZATION_METHOD = 'max_sharpe'  # 'max_sharpe', 'min_variance'
OPTIMIZATION_MAX_ITER = 2000
NO_SHORT = True
