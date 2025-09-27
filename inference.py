import logging
from churnexplainer import ExplainedModel

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_name: str):
        logger.info(f"Loading model: {model_name}")
        self.model = ExplainedModel(model_name)

    def predict(self, data):
        try:
            logger.info("Running predictions...")
            predictions, probabilities = self.model.predict_df(data)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def explain(self, data):
        try:
            logger.info("Generating explanation...")
            return self.model.explain(data)
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            raise
