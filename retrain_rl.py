"""Retrain only the RL agent with fixed environment."""
import os, sys, logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def send_telegram(msg):
    try:
        import requests
        requests.post('https://api.telegram.org/bot8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY/sendMessage',
            data={'chat_id': 7997570468, 'text': msg}, timeout=10)
    except: pass

send_telegram("RL retraining started! Fixes:\n- Bankruptcy protection (episode ends at -50%)\n- Entropy 0.15→0.08 (less wild)\n- P&L reward scaled 100→10 (less extreme)\n- 750K timesteps, fresh model\n\nXGB + LSTM are fine, keeping those.")

from src.config import get_config
from src.security.vault import SecureVault
from src.trading.okx_client import SecureOKXClient
from src.data.collector import DataCollector
from src.data.features import FeatureEngine, create_target_labels
from src.models.rl_agent import RLTradingAgent
import pandas as pd

config = get_config()
trading_pairs = config.trading.trading_pairs

vault = SecureVault()
okx = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)
collector = DataCollector(okx)
features = FeatureEngine()

logger.info("Collecting data...")
all_data = []
for pair in trading_pairs:
    try:
        df_1h = collector.get_historical_data(pair, days=180, timeframe="1h")
        df_4h = collector.get_historical_data(pair, days=180, timeframe="4h")
        df_1d = collector.get_historical_data(pair, days=180, timeframe="1d")
        df_f = features.calculate_multi_tf_features(df_1h, df_4h, df_1d)
        df_f["target"] = create_target_labels(df_f)
        df_f["pair"] = pair
        all_data.append(df_f)
        logger.info(f"  {pair}: {len(df_f)} samples")
    except Exception as e:
        logger.error(f"  {pair} failed: {e}")

combined_df = pd.concat(all_data, ignore_index=True).dropna()
feature_cols = features.get_feature_names()
logger.info(f"Total samples: {len(combined_df)}")

# Delete old RL model
for f in ["models/rl_agent.zip", "models/rl_agent_meta.pkl"]:
    if os.path.exists(f):
        os.remove(f)

t0 = time.time()
rl_agent = RLTradingAgent()
rl_agent.model_path = "models/rl_agent.zip"
rl_metrics = rl_agent.train(combined_df, feature_cols, total_timesteps=750000, initial_balance=1000)
elapsed = time.time() - t0

logger.info(f"RL Results: return={rl_metrics['total_return']:.2%}, portfolio=${rl_metrics['final_portfolio_value']:.2f}, trades={rl_metrics['total_trades']}")

ret = rl_metrics['total_return']
pval = rl_metrics['final_portfolio_value']
trades = rl_metrics['total_trades']

if pval > 0 and ret > -1.0:
    send_telegram(f"RL retrain SUCCESS!\n- Return: {ret:.2%}\n- Portfolio: ${pval:.2f}\n- Trades: {trades}\n- Time: {elapsed:.0f}s\n\nRestarting bot with new model...")
    os.system("sudo systemctl restart cryptobot")
    send_telegram("Bot restarted! Monitoring first signals...")
else:
    send_telegram(f"RL retrain still bad:\n- Return: {ret:.2%}\n- Portfolio: ${pval:.2f}\n- Trades: {trades}\n\nNOT restarting bot. Need more investigation.")

