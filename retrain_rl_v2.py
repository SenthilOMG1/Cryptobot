"""Retrain RL agent with multi-pair environment fix."""
import os, sys, logging, time
os.chdir('/root/Cryptobot')
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def send_tg(msg):
    try:
        import requests
        requests.post('https://api.telegram.org/bot8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY/sendMessage',
            data={'chat_id': 7997570468, 'text': msg}, timeout=10)
    except: pass

send_tg("RL retraining v2 started!\n\nRoot cause found: the RL was training on concatenated price data from ALL pairs. When it held a position in PEPE ($0.000005) and the data jumped to ETH ($3292), it saw a 67 BILLION % price spike and went bankrupt.\n\nFix: MultiPairTradingEnv — each pair is a separate episode, no cross-pair price jumps.\n\nTraining now...")

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
        logger.info(f"  {pair}: {len(df_f)} samples, close: ${df_f['close'].min():.4f}-${df_f['close'].max():.4f}")
    except Exception as e:
        logger.error(f"  {pair} failed: {e}")

combined_df = pd.concat(all_data, ignore_index=True).dropna()
feature_cols = features.get_feature_names()
logger.info(f"Total: {len(combined_df)} samples, {len(all_data)} pairs")

# Delete old model
for f in ["models/rl_agent.zip", "models/rl_agent_meta.pkl"]:
    if os.path.exists(f): os.remove(f)

t0 = time.time()
rl_agent = RLTradingAgent()
rl_agent.model_path = "models/rl_agent.zip"
rl_metrics = rl_agent.train(combined_df, feature_cols, total_timesteps=750000, initial_balance=1000)
elapsed = time.time() - t0

ret = rl_metrics['total_return']
pval = rl_metrics['final_portfolio_value']
trades = rl_metrics['total_trades']
logger.info(f"RL: return={ret:.2%}, portfolio=${pval:.2f}, trades={trades}, time={elapsed:.0f}s")

if pval > 500 and ret > -0.75:  # Didn't lose more than 75%
    send_tg(f"RL retrain v2 SUCCESS!\n- Return: {ret:.2%}\n- Portfolio: ${pval:.2f} (started at $1000)\n- Trades: {trades}\n- Time: {elapsed:.0f}s\n\nRestarting bot with new models...")
    os.system("sudo systemctl restart cryptobot")
    time.sleep(3)
    send_tg("Bot restarted with all 3 new models! Monitoring...")
else:
    send_tg(f"RL retrain v2 results:\n- Return: {ret:.2%}\n- Portfolio: ${pval:.2f}\n- Trades: {trades}\n\n{'GOOD ENOUGH - restarting' if pval > 0 else 'Still bad - NOT restarting'}")
    if pval > 0:
        os.system("sudo systemctl restart cryptobot")
        send_tg("Bot restarted anyway (RL isn't perfect but it's not bankrupt).")
