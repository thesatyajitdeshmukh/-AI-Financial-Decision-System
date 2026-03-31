from models.finbert_model import get_sentiment
from models.tft_model import predict_price
from models.gnn_model import get_market_signal
from models.ppo_model import get_decision
import torch

def run_pipeline():

    print("\n--- START PIPELINE ---")

    news = "Stock market is performing very well"
    market_data = [100, 102, 105, 110]
    graph_data = torch.randn(1, 10)

    sentiment = get_sentiment(news)
    predicted_price, trend = predict_price(market_data)
    market_signal = get_market_signal(graph_data)

    state = [sentiment, predicted_price, market_signal]

    decision, confidence = get_decision(state)

    print("--- END PIPELINE ---\n")

    return {
        "sentiment": sentiment,
        "prediction": predicted_price,
        "trend": trend,
        "market_signal": market_signal,
        "decision": decision,
        "confidence": confidence
    }