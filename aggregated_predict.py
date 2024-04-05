import helper
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


def aggregate_predict_by_score(models_predictions, models_metrics, models_weights, precision_weight=0.6, f1_weight=0.4):
    """
    Calculating the aggregated predictions using the weights of the models
    and their performance metrics, which means that each prediction is adjusted
    according to the importance (weights) and performance (weighted metrics) of the model.

    Parameters:
        models_predictions - A 2D array containing the arrays of predictions for each model.
        models_metrics - A 2D array containing the Precision and F1 for each model.
        models_weights - Array containing the weight of each model, the sum of which is 1.
        precision_weight - Floating point, the precision weight.
        f1_weight - Floating point, the f1 weight.

    Returns:
        Array of final predictions given after integrating all models.
    """

    # Calculate weighted scores using precision and f1
    weighted_scores = []
    for m in models_metrics:
        score = m[0] * precision_weight + m[1] * f1_weight
        weighted_scores.append(score)

    # The prediction results of each model are multiplied by its weights and weighted scores
    weighted_scores = np.array(weighted_scores)
    weighted_predictions = []
    for i in range(len(models_predictions)):
        weighted_pred = models_predictions[i] * models_weights[i] * weighted_scores[i]
        weighted_predictions.append(weighted_pred)

    '''
    The sum of the weighted predictions for all models is calculated and divided by 
    the sum of all model weights and weighted scores. This gives an average value that 
    reflects the combined predictions of all models.
    '''
    results = sum(weighted_predictions) / np.sum(models_weights * weighted_scores)

    # Converting predictions to binary classification results
    threshold = 0.5
    final_prediction = np.where(results >= threshold, 1, 0)

    return final_prediction


def aggregate_predict_by_vote(models_predictions):
    """
        Voting based on model predictions,
        if all sensor models and network models predict a result of 0, then the result is 0.
        Otherwise, the result is 1.

        Parameters:
            models_predictions - A 2D array containing the arrays of predictions for each model.

        Returns:
            Array of final predictions given after integrating all models.
    """

    majority_votes = []

    for i in range(len(models_predictions[0])):
        result_sum = 0
        for j in range(len(models_predictions) - 1):
            result_sum += models_predictions[j][i]

        if result_sum >= 1:
            vote = 1
        else:
            vote = 0
        majority_votes.append(vote)

    majority_votes = np.array(majority_votes)

    final_prediction = []
    for k in range(len(majority_votes)):
        if majority_votes[k] == models_predictions[-1][k]:
            final_prediction.append(majority_votes[k])
        else:
            final_prediction.append(1)
    final_prediction = np.array(final_prediction)

    return final_prediction


def get_predictions_and_metrics(local_models, sensor_test, global_model, network_test, roc=True):
    # Testing models using testing sets to obtain predictions and performance metrics

    sensor_x = sensor_test.iloc[:, :-1]
    sensor_y = sensor_test.iloc[:, -1]
    network_x = network_test.iloc[:, :-1]
    network_y = network_test.iloc[:, -1]

    models_metrics = []
    models_predictions = []
    fig, ax = plt.subplots()

    # Test the local models and show the metrics
    for i, model_path in enumerate(local_models):
        local_model = joblib.load(model_path)
        s_predict = local_model.predict(sensor_x)
        models_predictions.append(s_predict)

        print(f"Sensor Model {i + 1} Prediction Results:")
        accuracy, precision, recall, f1 = helper.get_metrics(sensor_y, s_predict, printout=True)
        models_metrics.append([precision, f1])

        if roc:
            # Draw ROC curve
            s_predict_proba = local_model.predict_proba(sensor_x)[:, 1]
            RocCurveDisplay.from_predictions(sensor_y, s_predict_proba, name=f'Sensor Model {i + 1}', ax=ax)

    # Test the global model and show the metrics
    network_model = joblib.load(global_model)
    n_predict = network_model.predict(network_x)
    models_predictions.append(n_predict)

    print(f"Network Model Prediction Results:")
    accuracy, precision, recall, f1 = helper.get_metrics(network_y, n_predict, printout=True)
    models_metrics.append([precision, f1])

    if roc:
        n_predict_proba = network_model.predict_proba(network_x)[:, 1]
        RocCurveDisplay.from_predictions(network_y, n_predict_proba, name='Network Model', ax=ax)
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend(loc="lower right")
        plt.show()

    return models_predictions, models_metrics
