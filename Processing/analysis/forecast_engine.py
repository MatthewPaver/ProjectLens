import pandas as pd
import numpy as np
# from statsmodels.tsa.arima.model import ARIMA # Remove fixed ARIMA
# import pmdarima as pm # Remove pmdarima import
from sklearn.preprocessing import MinMaxScaler
import logging
import traceback
import platform
import os
from datetime import datetime, timedelta

# Import necessary components from statsforecast
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# --- Re-enable TensorFlow/Keras ---
# Global flag to track if TensorFlow is available and stable
tensorflow_available = False # Will be set True in try block if successful
use_keras = False # Will be set True in try block if successful
# logging.warning("TensorFlow/Keras import explicitly disabled for debugging.") # Remove disable warning
# --- End Re-enable ---

# Safely import TensorFlow with fallback
try:
    # Configure TensorFlow for stability
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging noise
    
    # Special handling for Apple Silicon
    if platform.machine() in ["arm64", "aarch64"]:
        try:
            # Try the Apple-specific TensorFlow first
            import tensorflow_macos as tf
            logging.info("Using tensorflow_macos on Apple Silicon")
        except ImportError:
            import tensorflow as tf
            logging.info("Using standard tensorflow on Apple Silicon")
    else:
        import tensorflow as tf
    
    # Check if Metal plugin is available on Mac
    if platform.system() == 'Darwin':
        try:
            # Avoid loading GPU acceleration if it causes issues
            # tf.config.set_visible_devices([], 'GPU')
            pass
        except:
            logging.warning("Failed to configure GPU settings")
    
    # Import Keras components
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    
    # --- MODIFICATION: Keep Keras/TF disabled for speed ---
    tensorflow_available = False # Keep False
    use_keras = False # Keep False
    logging.info("TensorFlow/Keras imported BUT GANN forecasting is DISABLED for performance.")
    # --- END MODIFICATION ---
    
except ImportError:
    logging.warning("TensorFlow not available - will use statistical forecasting only") # Restore warning
    tensorflow_available = False
    use_keras = False
except Exception as e:
    logging.error(f"Error initializing TensorFlow: {e}") # Restore error logging
    logging.error(traceback.format_exc())
    tensorflow_available = False
    use_keras = False

def build_gann_model(input_shape):
    """Build a LSTM-based neural network forecasting model"""
    # --->>> ADD Logger instance <<<---
    logger = logging.getLogger(__name__)
    if not use_keras:
        logging.debug("Skipping GANN model build: use_keras is False.")
        return None
        
    try:
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # output forecast in UNIX timestamp scale
        model.compile(optimizer='adam', loss='mse')
        logging.debug("GANN model built successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building GANN model: {e}")
        return None

def forecast_gann(dates, task_id, project_name):
    """
    Neural network-based forecast with fallback to statistical methods.
    
    Args:
        dates: List of datetime objects representing historical dates
        task_id: Identifier for the task being processed (for logging).
        project_name: Name of the project (for logging).
        
    Returns:
        Tuple of (forecast_date, confidence_score)
    """
    if len(dates) < 3:
        logging.debug(f"[{project_name}-{task_id}] GANN: Not enough data points ({len(dates)} < 3), returning None.")
        return None, 0.0
        
    # Create fallback forecast using simple linear extrapolation
    def linear_extrapolate(dates):
        if len(dates) < 2:
            logging.debug(f"[{project_name}-{task_id}] GANN Fallback: Less than 2 dates, using last date + 7 days.")
            return dates[-1] + timedelta(days=7), 0.5
            
        # Calculate average time delta between dates
        deltas = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        if not deltas: # Handle case with only one delta if len(dates) == 2
             logging.debug(f"[{project_name}-{task_id}] GANN Fallback: Only one delta, using that delta.")
             avg_delta = deltas[0] if deltas else 7 # Default to 7 if somehow deltas is empty
        else:
            avg_delta = sum(deltas) / len(deltas)
        
        # Extrapolate next date
        next_date = dates[-1] + timedelta(days=avg_delta)
        
        # Calculate confidence based on consistency of deltas
        if len(deltas) > 1:
            std_dev = np.std(deltas)
            consistency = max(0, 1 - (std_dev / max(abs(avg_delta), 1)))
            confidence = min(0.8, max(0.4, consistency))
        else:
            confidence = 0.5
        logging.debug(f"[{project_name}-{task_id}] GANN Fallback: Linear extrapolation result: {next_date}, confidence: {confidence:.2f}")
        return next_date, confidence
    
    # If TensorFlow/Keras is not available, use the fallback method
    if not use_keras:
        # --->>> ADD Logger instance <<<---
        logger = logging.getLogger(__name__)
        logging.debug(f"[{project_name}-{task_id}] GANN: use_keras is False, using linear extrapolation fallback.")
        return linear_extrapolate(dates)
    
    # --->>> ADD Logger instance <<<---
    logger = logging.getLogger(__name__)
    try:
        logging.debug(f"[{project_name}-{task_id}] GANN: Attempting GANN forecast with {len(dates)} dates.")
        # Convert to UNIX timestamps
        unix_times = np.array([int(dt.timestamp()) for dt in dates])
        unix_times = unix_times.reshape(-1, 1)

        # Normalize
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(unix_times)

        # Prepare input for LSTM
        X = []
        y = []
        n_steps = 2 # Number of time steps to look back
        for i in range(len(scaled_data) - n_steps):
            X.append(scaled_data[i:i+n_steps])
            y.append(scaled_data[i+n_steps])
        X, y = np.array(X), np.array(y)

        # Reshape for LSTM input: [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        if len(X) < 1:
            logging.warning(f"[{project_name}-{task_id}] GANN: Not enough sequences created ({len(X)} < 1) after preparing LSTM input. Falling back.")
            return linear_extrapolate(dates)
            
        logging.debug(f"[{project_name}-{task_id}] GANN: Building model for input shape {X.shape}.")
        model = build_gann_model((X.shape[1], X.shape[2]))
        if model is None:
            logging.error(f"[{project_name}-{task_id}] GANN: Model building failed. Falling back.")
            return linear_extrapolate(dates)
            
        # Use shorter training to avoid potential crashes/long runs
        epochs = 15
        logging.debug(f"[{project_name}-{task_id}] GANN: Training model for {epochs} epochs...")
        model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)
        logging.debug(f"[{project_name}-{task_id}] GANN: Model training complete.")

        # Use last n_steps for prediction
        pred_input = scaled_data[-n_steps:].reshape((1, n_steps, 1))
        pred_scaled = model.predict(pred_input)[0][0]
        forecast_unix = scaler.inverse_transform([[pred_scaled]])[0][0]
        forecast_date = datetime.utcfromtimestamp(forecast_unix)
        loss = float(model.evaluate(X, y, verbose=0))
        confidence = max(0.7, min(0.98, 1.0 - loss)) # Confidence based on training loss
        logging.info(f"[{project_name}-{task_id}] GANN: Forecast successful. Date: {forecast_date}, Confidence: {confidence:.2f}")

        return forecast_date, round(confidence, 2)
        
    except Exception as e:
        logging.error(f"[{project_name}-{task_id}] GANN forecast error: {e}. Falling back to linear extrapolation.")
        logging.error(traceback.format_exc()) # Log traceback for GANN errors
        # Fall back to simple extrapolation if neural network fails
        return linear_extrapolate(dates)

def run_forecasting(df: pd.DataFrame, project_name: str) -> tuple[pd.DataFrame, list]:
    """
    Run time-series forecasting using StatsForecast AutoARIMA.
    
    Args:
        df: DataFrame with task_id, update_phase, and actual_finish columns.
        project_name: Name of the project for context.
        
    Returns:
        Tuple containing:
            - DataFrame with forecast results (successful forecasts).
            - List of task_ids that failed due to insufficient data.
    """
    # --->>> ADD Logger instance <<<---
    logger = logging.getLogger(__name__)
    logging.info(f"[{project_name}] Running forecasting engine using StatsForecast AutoARIMA...")
    results = []
    failed_tasks_insufficient_data = []
    
    if df.empty:
        logging.warning(f"[{project_name}] Forecasting: Input DataFrame is empty.")
        return pd.DataFrame(), []
        
    # --->>> Log Input Details <<<---
    logger.info(f"[{project_name}] Forecasting: Input shape={df.shape}, Columns={df.columns.tolist()}")
    # --->>> END Log Input Details <<<---

    # Validate required columns - use standardised names
    required_cols = ["task_id", "update_phase", "actual_finish"] # task_name no longer strictly needed here
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"[{project_name}] Forecasting: Missing required columns: {missing_cols}")
        return pd.DataFrame(), []
    # --->>> Log Column Check Pass <<<---
    logger.debug(f"[{project_name}] Forecasting: Required columns {required_cols} are present.")
    # --->>> END Log Column Check Pass <<<---

    # Prepare DataFrame for StatsForecast: needs unique_id, ds, y
    try:
        # Convert actual_finish to datetime and drop NaNs *before* selecting columns
        # --->>> Log Date Conversion <<<---
        logger.debug(f"[{project_name}] Forecasting: Converting 'actual_finish' to datetime and handling errors...")
        df['actual_finish_dt'] = pd.to_datetime(df['actual_finish'], errors='coerce')
        pre_drop_len = len(df)
        sf_df = df.dropna(subset=['actual_finish_dt'])
        post_drop_len = len(sf_df)
        logger.debug(f"[{project_name}] Forecasting: Dropped {pre_drop_len - post_drop_len} rows with invalid 'actual_finish' dates.")
        # --->>> END Log Date Conversion <<<---

        if sf_df.empty:
            logging.warning(f"[{project_name}] Forecasting: No valid actual_finish dates after conversion and NaN drop.")
            return pd.DataFrame(), list(df['task_id'].unique()) # All tasks failed
            
        # Check minimum data points *per task*
        # --->>> Log Data Point Check <<<---
        min_data_points = 3
        logger.debug(f"[{project_name}] Forecasting: Checking for tasks with >= {min_data_points} valid data points...")
        task_counts = sf_df['task_id'].value_counts()
        valid_tasks = task_counts[task_counts >= min_data_points].index.tolist()
        invalid_tasks = task_counts[task_counts < min_data_points].index.tolist()
        failed_tasks_insufficient_data.extend(invalid_tasks)
        logger.info(f"[{project_name}] Forecasting: Found {len(valid_tasks)} tasks with sufficient data and {len(invalid_tasks)} tasks with insufficient data.")
        if invalid_tasks:
             logger.debug(f"[{project_name}] Tasks with insufficient data: {invalid_tasks[:10]}...") # Log first few failed tasks
        # --->>> END Log Data Point Check <<<---

        if not valid_tasks:
            logging.warning(f"[{project_name}] Forecasting: No tasks have sufficient data points (>= {min_data_points}) after cleaning.")
            return pd.DataFrame(), failed_tasks_insufficient_data
            
        # Filter sf_df to include only tasks with enough data
        sf_df = sf_df[sf_df['task_id'].isin(valid_tasks)].copy() 
        # --->>> Log Filtering <<<---
        logger.debug(f"[{project_name}] Forecasting: Filtered DataFrame shape for valid tasks: {sf_df.shape}")
        # --->>> END Log Filtering <<<---
        
        # Rename columns and select required ones
        sf_df = sf_df.rename(columns={
            'task_id': 'unique_id',
            'actual_finish_dt': 'ds' 
            # We need a numeric value 'y'. Let's use the timestamp.
        })
        # Convert datetime 'ds' to timestamp for 'y'
        sf_df['y'] = sf_df['ds'].apply(lambda x: x.timestamp())
        
        # Keep only the necessary columns for statsforecast input
        sf_df = sf_df[['unique_id', 'ds', 'y']]
        
        # Sort by id and date as required by statsforecast
        sf_df = sf_df.sort_values(by=['unique_id', 'ds'])
        
        logger.info(f"[{project_name}] Forecasting: Prepared DataFrame for StatsForecast with {len(valid_tasks)} tasks having >={min_data_points} points.")
        logger.debug(f"StatsForecast input DataFrame head:\n{sf_df.head().to_string()}")
        
    except Exception as e:
        logger.error(f"[{project_name}] Error preparing data for StatsForecast: {e}", exc_info=True)
        return pd.DataFrame(), list(df['task_id'].unique()) # Assume all failed if prep fails

    # Run forecasting using StatsForecast
    try:
        # Define the model
        models = [AutoARIMA()] # Use AutoARIMA from statsforecast
        
        # Instantiate StatsForecast
        # Frequency 'D' for daily. Adjust if data suggests otherwise.
        # n_jobs=-1 uses all available cores
        sf = StatsForecast(
            models=models,
            freq='D', 
            n_jobs=-1
        )
        
        # Fit the model to the historical data
        logger.info(f"[{project_name}] Fitting StatsForecast models...")
        sf.fit(sf_df)
        logger.debug(f"[{project_name}] Forecasting: Fit completed.")
        
        # Make predictions - horizon 'h' needs to be an integer
        forecast_df = sf.predict(h=1, level=[95]) # Predict 1 step ahead with 95% confidence
        logger.debug(f"[{project_name}] Forecasting: Predict completed.")
        
        # Convert timestamp forecast back to datetime
        forecast_df['forecast_date'] = pd.to_datetime(forecast_df['AutoARIMA'], unit='s')
        
        # Add forecast confidence (derived from prediction interval width)
        # A narrower interval implies higher confidence
        interval_width = forecast_df['AutoARIMA-hi-95'] - forecast_df['AutoARIMA-lo-95']
        # Scale interval width relative to the forecast value to get a relative measure
        # Avoid division by zero if forecast is 0
        forecast_df['relative_interval_width'] = interval_width / (forecast_df['AutoARIMA'].abs().replace(0, 1))
        # Map relative width to confidence (e.g., smaller width = higher confidence)
        # This mapping is heuristic and might need adjustment
        forecast_df['forecast_confidence'] = (1 / (1 + forecast_df['relative_interval_width'])).clip(0.5, 0.98) # Bounded between 0.5 and 0.98

        # --->>> ADDED: Include model type and low confidence flag <<<---
        forecast_df['model_type'] = 'AutoARIMA (statsforecast)'
        low_confidence_threshold = 0.75
        forecast_df['low_confidence_flag'] = forecast_df['forecast_confidence'] < low_confidence_threshold
        # --->>> END ADDED <<<---

        # --->>> CHANGE: Merge results back - include task_name, rename date <<<---
        # Select necessary columns from the forecast
        forecast_output = forecast_df[['unique_id', 'AutoARIMA', 'forecast_confidence', 'model_type', 'low_confidence_flag']].copy()
        # Convert forecast timestamp to datetime and rename
        forecast_output['predicted_end_date'] = pd.to_datetime(forecast_output['AutoARIMA'], unit='s')
        
        # Prepare original df for merge (get latest task_name per task_id)
        # Ensure we handle the case where the input df might not have update_phase
        if 'update_phase' in df.columns:
            latest_tasks = df.sort_values('update_phase', ascending=False).drop_duplicates(subset=['task_id'])[['task_id', 'task_name']]
        else:
            # Fallback if update_phase is missing: just drop duplicates, hoping the last occurrence is the latest name
            latest_tasks = df[['task_id', 'task_name']].drop_duplicates(subset=['task_id'], keep='last')
            
        # Merge task names into the forecast output
        results_df = pd.merge(forecast_output, latest_tasks, left_on='unique_id', right_on='task_id', how='left')

        # Select and rename final columns
        results_df = results_df[['task_id', 'task_name', 'predicted_end_date', 'forecast_confidence', 'model_type', 'low_confidence_flag']]
        # --->>> END CHANGE <<<---

        if not results_df.empty:
             # Convert dates back to datetime objects if needed for consistency
             results_df['predicted_end_date'] = pd.to_datetime(results_df['predicted_end_date'])
             logger.info(f"[{project_name}] Forecasting completed. Generated {len(results_df)} forecasts using StatsForecast.")
        else:
             logger.warning(f"[{project_name}] Forecasting completed, but no valid forecast results generated by StatsForecast.")

        final_shape = results_df.shape if isinstance(results_df, pd.DataFrame) else (0, 0)
        logger.info(f"[{project_name}] Forecasting engine finished. Returning DataFrame with shape: {final_shape}")
        if final_shape[0] == 0:
            logger.warning(f"[{project_name}] StatsForecast produced NO results. Check input data history and logs.")

        return results_df, failed_tasks_insufficient_data

    except Exception as e:
        logger.error(f"[{project_name}] Error during StatsForecast fitting/prediction: {e}", exc_info=True)
        # Return empty results, keep previously identified failed tasks
        return pd.DataFrame(), failed_tasks_insufficient_data

def create_sequences(data, n_steps):
    """Create sequences for LSTM model"""
    X = []
    y = []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)