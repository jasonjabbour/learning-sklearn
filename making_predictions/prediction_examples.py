from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_blobs, make_regression
import pickle
import os

MODEL_DIRECTORY = 'models'
TEST_SIZE = .33
SEED = 7
logistic_regression_model = None
linear_regression_model = None

def create_logistic_regression_model():
    '''Logistic Regression Model Example'''

    # Generate 2D Classification Dataset
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=SEED)

    # Test and Train Split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y,
                                                                        test_size=TEST_SIZE, 
                                                                        random_state=SEED)

    # Fit Final Model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    #Score model
    print(f'Logisitc Regression Model Score: {model.score(X_test, Y_test)}')

    return model

def logistic_regression_predict(model):
    '''Make a prediction using the logistic regression model'''

    # Get instance where you do not know the answer
    Xnew, y = make_blobs(n_samples=3, centers=2, n_features=2, random_state=SEED)

    # Class Prediction
    ynew_class = model.predict(Xnew)

    # Probability Prediction of belonging to a class
    ynew_probability = model.predict_proba(Xnew)

    # Show the inputs and predicted outputs
    for i in range(len(Xnew)):
        # Print Class Prediciton
        print('Logistic Regression Class Predictions: ')
        print('X=%s, Predicted=%s, Y=%s' % (Xnew[i], ynew_class[i], y[i]))

        # Print Probability Prediciton
        print('Logistic Regression Probability Predictions: ')
        print('X=%s, Predicted=%s, Y=%s \n' % (Xnew[i], ynew_probability[i], y[i]))

def create_linear_regression_model():
    '''Create a linear regression model'''
    
    # Generate 2D Classification Dataset
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    # Test and Train Split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y,
                                                                        test_size=TEST_SIZE, 
                                                                        random_state=SEED)

    # Fit Final Model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    #Score model
    print(f'Linear Regression Model Score: {model.score(X_test, Y_test)}')

    return model


def linear_regression_predict(model):
    '''Make a prediction using the linear regression model'''

    # Get instance where you do not know the answer
    Xnew, y = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=SEED)

    # Class Prediction
    ynew = model.predict(Xnew)

    # Show the inputs and predicted outputs
    for i in range(len(Xnew)):
        # Print Prediciton
        print('Linear Regression Predictions: ')
        print('X=%s, Predicted=%s, Y=%s' % (Xnew[i], ynew[i], y[i]))


def save_model(model, filename):
    #Save the model using pickle to serialize ML model
    path = os.path.join(MODEL_DIRECTORY, filename)
    pickle.dump(model, open(path, 'wb'))
    print(f'Model saved: {path}')

def load_model(filename):
    '''Load model from computer'''
    path = os.path.join(MODEL_DIRECTORY, filename)
    loaded_model = pickle.load(open(path, 'rb'))
    print('Model Loaded!')
    return loaded_model

def print_algo_menu(algorithm_name):
    '''Print Menu'''
    print('\n')
    print(f'--------- {algorithm_name} Menu ---------')
    print(f'[1]\tTrain {algorithm_name} Model')
    print(f'[2]\tSave {algorithm_name} Model')
    print(f'[3]\tLoad {algorithm_name} Model')
    print(f'[4]\tMake Prediction using {algorithm_name} Model')
    print(f'[leave]\tLeave {algorithm_name} Menu')
    print(f'[exit]\tExit Program')

def navigate_algo_menu(model, mode_file_name, create_model=None, predict_model=None):
    user_input = input("Select choice: ")
    print('\n')

    # Training Logistic Regression Model     
    if user_input == '1': 
        model = create_model()

    # Saving Regression Model
    elif user_input == '2':
        save_model(model, mode_file_name)

    # Load Logistic Regression Model
    elif user_input == '3':
        model = load_model(mode_file_name)

    # Make prediction using Logistic Regression Model
    elif user_input == '4':
        if model == None: 
            print("Please train or load a model first.")

        predict_model(model)
    
    # Leave Logistic Regression Menu
    elif user_input.lower() == 'leave':
        return model, 'leave' 
    
    # End Program
    elif user_input.lower() == 'exit':
        return model, 'exit'

    return model, True

def logistic_regression_menu():
    '''Logistic Regression Menu'''
    global logistic_regression_model

    algorithm_name = 'Logistic Regression'
    logistic_regression_model_name = 'logistic_regression_model_example1.sav'

    while True:
        # Logistic Regression Menu
        print_algo_menu(algorithm_name)
        # Navigate Algo Menu 
        logistic_regression_model, next_step = navigate_algo_menu(logistic_regression_model,
                                    logistic_regression_model_name, 
                                    create_model = create_logistic_regression_model, 
                                    predict_model = logistic_regression_predict)

        if next_step == 'leave':
            return True
        elif next_step == 'exit':
            return False

def linear_regression_menu():
    '''Linear Regression Menu'''
    global linear_regression_model

    algorithm_name = 'Linear Regression'
    linear_regression_model_name = 'linear_regression_model_example1.sav'

    while True:
        # Logistic Regression Menu
        print_algo_menu(algorithm_name)
        # Navigate Algo Menu 
        linear_regression_model, next_step = navigate_algo_menu(linear_regression_model,
                                    linear_regression_model_name, 
                                    create_model = create_linear_regression_model, 
                                    predict_model = linear_regression_predict)

        if next_step == 'leave':
            return True
        elif next_step == 'exit':
            return False



if __name__ == '__main__':

    while True:
        print('\n')
        print('--------- Menu --------')
        print('[1] Logistic Regression')
        print('[2] Linear Regression')
        print('[exit] Exit Program')
        print('--------- Menu --------')
        user_input = input("Select choice: ")
        print('\n')

        if user_input == '1':
            next_step = logistic_regression_menu()
        elif user_input == '2':
            next_step = linear_regression_menu()
        elif user_input.lower() == 'exit':
            break

        if not next_step:
            break


