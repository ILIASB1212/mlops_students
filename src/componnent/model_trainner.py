import os
import sys
from dataclasses import dataclass
from xml.parsers.expat import model

from catboost import CatBoostRegressor
from sklearn.ensemble import (
AdaBoostRegressor,
GradientBoostingRegressor,
RandomForestRegressor,)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.costum_expection import CustomException
from src.loger import get_logger
from src.utils import save_object, evaluate_models
from sklearn.metrics import r2_score

logger=get_logger(__name__)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array,preprossessor_path=None):
        try:
            logger.info("splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            logger.info("splitting training and testing input data completed")
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test =y_test,models=models)
                        ## To get best model score from diet
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_mame = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_mame]
            logger.info(f"best model found on both training and testing dataset {best_model_mame} with r2 score: {best_model_score}")
            if best_model_score < 0.6:
                logger.info("No best model found")
                raise CustomException("No best model found")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            logger.error("Error occured in Data Transformation method ")
            raise CustomException(f"error occurred in data transformation method: {str(e)}", sys)
        