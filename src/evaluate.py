
from models import rmse, global_mean_model, user_mean_model, item_mean_model


def evaluate_models(train, test):
    actual = test["rating"].to_numpy()

    models = {
        "Global Mean": global_mean_model(train, test),
        "User Mean": user_mean_model(train, test),
        "Movie Mean": item_mean_model(train, test),
    }

    for name, predicted in models.items():
        print(name, "RMSE:", rmse(actual, predicted))
