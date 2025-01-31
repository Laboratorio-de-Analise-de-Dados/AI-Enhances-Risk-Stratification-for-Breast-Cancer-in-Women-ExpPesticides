# Importing the pandas library for data manipulation
import pandas as pd

# Importing the numpy library for efficient numeric operations
import numpy as np

# Importing of SMOTE technique to deal with imbalanced data
from imblearn.over_sampling import SMOTE

# Import types needed for the annotation
from typing import List, Any

# Importing the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Importing the Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

# Importing the Support Vector Machine classifier
from sklearn.svm import SVC

# Importing the Multivariate Regression classifier
from sklearn.linear_model import LogisticRegression

# Importing the train_valid_test_split function from the fast_ml library for data splitting
from fast_ml.model_development import train_valid_test_split

# Importing the GridSearchCV function from scikit-learn for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Importing classification performance metrics from scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Importing the statistics library for basic statistical functions
import statistics as st

# Importing the joblib library for saving and loading models
import joblib 

# Importing the scikitplot library for visualizing model performance plots
import scikitplot as skplt

# Importing the classification_report function from scikit-learn for classification reports
from sklearn.metrics import classification_report

# Importing the confusion_matrix function from scikit-learn for the confusion matrix
from sklearn.metrics import confusion_matrix

# Importing the seaborn library for data visualization
import seaborn as sn

# Importing the matplotlib library for creating plots
import matplotlib.pyplot as plt

# Importing the roc_curve and auc functions from scikit-learn for ROC curve calculation and AUC
from sklearn.metrics import roc_curve, auc

class ModelosAlgoritmos:
    
    def __init__(self, dados, predicao, variaveis, oversample ,titulo_plot, titulo_geral):
        self.df = pd.read_excel(dados, index_col='Número')
        self.predicao = predicao
        self.variaveis = variaveis
        self.oversample = oversample
        self.titulo_plot = titulo_plot
        self.titulo_geral = titulo_geral
    
    def estrutura_estratifica(self):
        df_2 = self.df.loc[:, self.variaveis]
        df_3 = df_2.dropna()
        
        
        if self.oversample == 'Sim':
            # Separating the features and the target variable
            X = df_3.drop("Risk stratification", axis=1)
            y = df_3["Risk stratification"]
            
        try:
            # Tentar aplicar o SMOTE com os parâmetros padrão
            smote = SMOTE(sampling_strategy="auto")
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
        except ValueError as e:
            # Se ocorrer um ValueError, ajustar os parâmetros e tentar novamente
            smote = SMOTE(k_neighbors=min(3, len(X)-1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
                
            dados = pd.merge(X_resampled, y_resampled, left_index = True, right_index = True, how = 'inner')
            
        else:
            df_2 = self.df.loc[:, self.variaveis]
            dados = df_2.dropna()
            
        return dados
    
    
    def estrutura_exposicao(self):
        df_2 = self.df.loc[:, self.variaveis]
        df_3 = df_2.dropna()

        if 'Risk stratification' in df_3.columns:
            df_3['Risk stratification'].replace('Medium', 2, inplace=True)
            df_3['Risk stratification'].replace('High', 3, inplace=True)
            df_3['Risk stratification'].replace('Low', 1, inplace=True)

        
        
        if self.oversample == 'Sim':
            # Separating the features and the target variable
            X = df_3.drop("Exposure to pesticides", axis=1)
            y = df_3["Exposure to pesticides"]

            # Applying Oversampling to the entire dataset
            smote = SMOTE(sampling_strategy="auto")
            X_resampled, y_resampled = smote.fit_resample(X, y)

            dados = pd.merge(X_resampled, y_resampled, left_index = True, right_index = True, how = 'inner')
        
        else:  
            valor = df_3['Exposure to pesticides'].value_counts()
            valor_a = valor.tolist()
            menor_valor = min(valor_a)

            classe_0 = df_3[df_3['Exposure to pesticides'] == 0]
            classe_1 = df_3[df_3['Exposure to pesticides'] == 1]

            amostras_classe_0 = classe_0.sample(n=menor_valor, replace=True)
            amostras_classe_1 = classe_1.sample(n=menor_valor, replace=True)

            dados_balanceados = pd.concat([amostras_classe_0, amostras_classe_1])
            dados = dados_balanceados.sample(frac=1, random_state=42).reset_index(drop=True)
            dados = dados.dropna()

        return dados

    
    def verificar_positividade(self, row):
        if row['Chemoresistance'] == 1 or row['Recurrence'] == 1 or row['Death'] == 1:
            return 1  # positive
        else:
            return 0  # negative
    
    
    
    def estrutura_pior_prognostico(self):
        self.df['Worst prognosis'] = self.df.apply(self.verificar_positividade, axis=1)
        df_2 = self.df.loc[:, self.variaveis]
        df_3 = df_2.dropna()

        if 'Risk stratification' in df_3.columns:
            df_3['Risk stratification'].replace('Medium', 2, inplace=True)
            df_3['Risk stratification'].replace('High', 3, inplace=True)
            df_3['Risk stratification'].replace('Low', 1, inplace=True)

        
        if self.oversample == 'Sim':
            # Separating the features and the target variable
            X = df_3.drop("Worst prognosis", axis=1)
            y = df_3["Worst prognosis"]

            # Applying Oversampling to the entire dataset
            smote = SMOTE(sampling_strategy="auto")
            X_resampled, y_resampled = smote.fit_resample(X, y)

            dados = pd.merge(X_resampled, y_resampled, left_index = True, right_index = True, how = 'inner')
        
        else:
            valor = df_3['Worst prognosis'].value_counts()
            valor_a = valor.tolist()
            menor_valor = min(valor_a)

            # Splitting the DataFrame into two classes
            classe_0 = df_3[df_3['Worst prognosis'] == 0]
            classe_1 = df_3[df_3['Worst prognosis'] == 1]

            # Randomly selecting 94 samples from each class
            amostras_classe_0 = classe_0.sample(n=menor_valor, replace=True)
            amostras_classe_1 = classe_1.sample(n=menor_valor, replace=True)

            # Joining the selected samples
            dados_balanceados = pd.concat([amostras_classe_0, amostras_classe_1])

            # Reshuffling the data to ensure random order
            dados = dados_balanceados.sample(frac=1, random_state=42).reset_index(drop=True)


        return dados
        

    def estrutura_dados(self):

        if self.predicao == 'Risk stratification':
            self.variaveis.append(self.predicao)
            dados_estruturados = self.estrutura_estratifica()
            return dados_estruturados
        elif self.predicao == 'Exposure to pesticides':
            self.variaveis.append(self.predicao)
            dados_estruturados = self.estrutura_exposicao()
            return dados_estruturados
        elif self.predicao == 'Worst prognosis':
            self.variaveis.append(self.predicao)
            dados_estruturados = self.estrutura_pior_prognostico()
            return dados_estruturados
        else:
            return None
        
    def informacoes_df(self, dados):
        
        tamanho = len(dados)
        divisao_classes = dados[self.predicao].value_counts().to_dict()
        informacoes = [tamanho, divisao_classes]
        nome = 'informacoes_df_'+self.titulo_geral
        joblib.dump(informacoes, nome)
        
        return informacoes
        
    def grid_search_RL(self,dados: pd.DataFrame):
    
        ''' Selecting the best Logistic Regression model hyperparameters for my dataset using GridSearchCV '''
    
        # Dividing the dataset into training (60%), testing (30%) and validation (10%)
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(dados, target = self.predicao, 
                                                                                train_size=0.60, valid_size=0.1, test_size=0.30)

        # Selecting the hyperparameters of interest for testing
        parameters_dictionary = {
            'penalty':  ['l1', 'l2'],
            'solver': ['liblinear', 'Newton-CG'],
        }                        

        # Defining the prediction model
        modelo = LogisticRegression()

        # Carrying out testing on GridSearchCV
        grid_search = GridSearchCV(modelo, 
                                   parameters_dictionary, 
                                   return_train_score=True, 
                                   cv=5,
                                   verbose=1) 
        grid_search.fit(X_train, y_train)

        # Defining the best model
        best_model = grid_search.best_estimator_
        
        joblib.dump(best_model, 'melhor_modelo_RL')


        print('The best model was:', best_model)
        return best_model
    
    def grid_search_RF(self,dados: pd.DataFrame):
    
        ''' Selecting the best Random Forest model hyperparameters for my dataset using GridSearchCV '''
    
        # Dividing the dataset into training (60%), testing (30%) and validation (10%)
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(dados, target = self.predicao, 
                                                                                train_size=0.60, valid_size=0.1, test_size=0.30)

        # Defining the prediction model
        modelo = RandomForestClassifier()

        # Selecting the hyperparameters of interest for testing
        parameters_dictionary = {
            'criterion':  ['gini', 'entropy'],
            'n_estimators': [100, 200, 400],
            'max_depth':  [None, 4, 10],
            'min_samples_split': [1,5,15],
            'max_features': ['log2', 'float'],
            'class_weight': ['balanced', 'balanced_subsample']
        }                     

        # Carrying out testing on GridSearchCV
        grid_search = GridSearchCV(modelo, 
                                   parameters_dictionary, 
                                   return_train_score=True, 
                                   cv=5,
                                   verbose=1) 
        grid_search.fit(X_train, y_train)

        # Defining the best model
        best_model = grid_search.best_estimator_
        
        joblib.dump(best_model, 'melhor_modelo_RF')


        print('The best model was:', best_model)
        return best_model
    
    def grid_search_SVM(self,dados: pd.DataFrame):
    
        ''' Selecting the best SVM model hyperparameters for my dataset using GridSearchCV '''
    
        # Dividing the dataset into training (60%), testing (30%) and validation (10%)
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(dados, target = self.predicao, train_size=0.60, valid_size=0.1, test_size=0.30)

        # Selecting the hyperparameters of interest for testing
        parameters_dictionary = {'kernel':['linear', 'rbf'], 
                                 'C':[0.001, 0.01, 1, 10, 100], 
                                 'gamma':[1, 10, 100, 1000]}

        # Defining the prediction model
        modelo = SVC()

        # Carrying out testing on GridSearchCV
        grid_search = GridSearchCV(modelo, 
                                   parameters_dictionary, 
                                   return_train_score=True, 
                                   cv=5,
                                   verbose=1) 
        grid_search.fit(X_train, y_train)

        # Defining the best model
        best_model = grid_search.best_estimator_
        
        joblib.dump(best_model, 'melhor_modelo_SVM')


        print('The best model was:', best_model)
        return best_model
    
    def grid_search_GBOOST(self,dados: pd.DataFrame):
    
        ''' Selecting the best GBoost model hyperparameters for my dataset using GridSearchCV '''
    
        # Dividing the dataset into training (60%), testing (30%) and validation (10%)
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(dados, target = self.predicao, 
                                                                                train_size=0.60, valid_size=0.1, test_size=0.30)

        # Selecting the hyperparameters of interest for testing
        parameters_dictionary = {
            'n_estimators': [50, 100],
            'max_depth': [3, 4],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }                        

        # Defining the prediction model
        modelo = GradientBoostingClassifier()

        # Carrying out testing on GridSearchCV
        grid_search = GridSearchCV(modelo, 
                                   parameters_dictionary, 
                                   return_train_score=True, 
                                   cv=5,
                                   verbose=1) 
        grid_search.fit(X_train, y_train)

        # Defining the best model
        best_model = grid_search.best_estimator_
        
        joblib.dump(best_model, 'melhor_modelo_GBoost')


        print('The best model was:', best_model)
        return best_model
    
    
    ''' Defining functions for testing and evaluating model performance '''
    
    def cria_metricas(self,dados: pd.DataFrame, melhor_modelo: list):
    
        ''' Training the model and analyzing the evaluation metrics obtained with the prediction '''
    
        # Dividing the dataset into training (60%), testing (30%) and validation (10%)
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(dados, target = self.predicao, 
                                                                                train_size=0.60, valid_size=0.1, test_size=0.30)
        
        X_test = pd.concat([X_valid, X_test], axis=0)
        y_test = pd.concat([y_valid, y_test], axis=0)

        # Defining the prediction model
        modelo = melhor_modelo
        # Performing model training and testing
        modelo_treinado = modelo.fit(X_train, y_train)  
        previsoes = modelo_treinado.predict(X_test)

        # Calculating evaluation metrics
        acuracia = accuracy_score(y_test, previsoes) * 100
        precisao = precision_score(y_test, previsoes, average='weighted') * 100
        recall = recall_score(y_test, previsoes, average='weighted') * 100
        f1= (f1_score(y_test, previsoes, average='weighted') * 100)


        return acuracia, precisao, recall, f1
    
    
    def calculo_metricas(self,dados: pd.DataFrame, melhor_modelo: list, nome_tabela: str):

        
        lista_acuracias = []
        lista_precisao = []
        lista_recall = []
        lista_f1 = []

        for _ in range(100):
            acuracia, precisao, recall, f1 = self.cria_metricas(dados, melhor_modelo)
            lista_acuracias.append(acuracia)
            lista_precisao.append(precisao)
            lista_recall.append(recall)
            lista_f1.append(f1)

        dic_metricas = {
                'acurácia' : lista_acuracias,
                'precisao' : lista_precisao,
                'recall' : lista_recall,
                'f1_score' : lista_f1
        }

        tabela_metricas = pd.DataFrame(data = dic_metricas)
        tabela_metricas.columns = ['Accuracy', 'Precision', 'Recall', 'F1-score']

        # Saving the DataFrame containing the metrics
        joblib.dump(tabela_metricas, nome_tabela)

        return tabela_metricas
    
    
    def matrix_confusao(self, y_test: pd.DataFrame, previsoes: List[str], nome_matrix: str) -> None:

        ''' Constructing confusion matrix and relative frequency matrix '''

        # Identifying classes
        classes = sorted(set(y_test))

        # Confusion matrix
        cm = confusion_matrix(y_test, previsoes)
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        print(df_cm)

        # Creating a dictionary to store the expected values of class accuracy
        expected_values = {}

        # Calculating expected values (TP + FP) for each class and storing them in the dictionary
        for i, class_name in enumerate(classes):
            expected_value = sum(cm[i, :])
            expected_values[class_name] = expected_value

        # Creating variables to store the expected values for each class
        expected_value_0 = expected_values[0.0]
        expected_value_1 = expected_values[1.0]

        # Creating the structure of the ternary matrix
        conf_matrix_tern = pd.DataFrame(
            index=["Pred_Exposed", "Pred_Unexposed"],
            columns=["Obs_Exposed", "Obs_Unexposed"],
            data=np.nan
        )

        # Calculating the relative frequency
        conf_matrix_tern.iloc[0, 0] = round((cm[0, 0] / expected_value_0) * 100, 2)
        conf_matrix_tern.iloc[0, 1] = round((cm[0, 1] / expected_value_0) * 100, 2)


        conf_matrix_tern.iloc[1, 0] = round((cm[1, 0] / expected_value_1) * 100, 2)
        conf_matrix_tern.iloc[1, 1] = round((cm[1, 1] / expected_value_1) * 100, 2)


        print()

        # Plotting the relative frequency confusion matrix

        # Create a custom color map
        cmap = plt.cm.get_cmap('magma')
        
        # Create the figure instance
        fig = plt.figure()

        # Create the color plot
        plt.imshow(conf_matrix_tern, cmap=cmap, vmin=0, vmax=100)

        # Add cell values to the plot
        for i in range(conf_matrix_tern.shape[0]):
            for j in range(conf_matrix_tern.shape[1]):
                plt.text(j, i, f'{conf_matrix_tern.iloc[i, j]:.2f}', ha='center', va='center', color='w', fontsize=12)

        # Add a color bar to indicate value-to-color mapping
        plt.colorbar(label='Intensity')

        # Set labels for the X and Y axes, if desired
        plt.xticks(range(conf_matrix_tern.shape[1]), classes)
        plt.yticks(range(conf_matrix_tern.shape[0]), classes)

        # Show the plot
        plt.title("Relative Frequency Confusion Matrix")
        plt.show()
        
        # Adding  general title
        fig.suptitle(nome_matrix)
        
        # Saving the graph
        fig.savefig(f'{nome_matrix}.png')
    
    def matrix_confusao_ternaria(self, y_test: pd.DataFrame, previsoes: List[str], nome_matrix: str) -> None:
    
        ''' Constructing confusion matrix and relative frequency matrix '''

        # Identifying classes
        classes = sorted(set(y_test))

        # Confusion matrix
        cm = confusion_matrix(y_test, previsoes)
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        print(df_cm)

        # Creating a dictionary to store the expected values of class accuracy
        expected_values = {}

        # Calculating expected values (TP + FP) for each class and storing them in the dictionary
        for i, class_name in enumerate(classes):
            expected_value = sum(cm[i, :])
            expected_values[class_name] = expected_value

        # Creating variables to store the expected values for each class
        if 2 in expected_values:
            expected_value_high = expected_values[2]
        elif 'High' in expected_values:
            expected_value_high = expected_values['High']
        else:
            expected_value_high = 0

        if 1 in expected_values:
            expected_value_medium = expected_values[1]
        if 'Medium' in expected_values:
            expected_value_medium = expected_values['Medium']
        else:
            expected_value_medium = 0

        if 0 in expected_values:
            expected_value_low = expected_values[0]
        if 'Low' in expected_values:
            expected_value_low = expected_values['Low']
        else:
            expected_value_low = 0

        # Creating the structure of the ternary matrix
        conf_matrix_tern = pd.DataFrame(
            index=["Pred_High", "Pred_Medium", "Pred_Low"],
            columns=["Obs_High", "Obs_Medium", "Obs_Low"],
            data=np.nan
        )

        # Calculating the relative frequency
        if expected_value_high > 0:
            conf_matrix_tern.iloc[0, 0] = round((cm[0, 0] / expected_value_high) * 100, 2)
            conf_matrix_tern.iloc[0, 1] = round((cm[0, 1] / expected_value_high) * 100, 2)
            conf_matrix_tern.iloc[0, 2] = round((cm[0, 2] / expected_value_high) * 100, 2)

        if expected_value_medium > 0:
            conf_matrix_tern.iloc[1, 0] = round((cm[1, 0] / expected_value_medium) * 100, 2)
            conf_matrix_tern.iloc[1, 1] = round((cm[1, 1] / expected_value_medium) * 100, 2)
            conf_matrix_tern.iloc[1, 2] = round((cm[1, 2] / expected_value_medium) * 100, 2)

        if expected_value_low > 0:
            conf_matrix_tern.iloc[2, 0] = round((cm[2, 0] / expected_value_low) * 100, 2)
            conf_matrix_tern.iloc[2, 1] = round((cm[2, 1] / expected_value_low) * 100, 2)
            conf_matrix_tern.iloc[2, 2] = round((cm[2, 2] / expected_value_low) * 100, 2)

        print()

        # Plotting the relative frequency confusion matrix

        # Create a custom color map
        cmap = plt.cm.get_cmap('magma')
        
        # Create the figure instance
        fig = plt.figure()

        # Create the color plot
        plt.imshow(conf_matrix_tern, cmap=cmap, vmin=0, vmax=100)

        # Add cell values to the plot
        for i in range(conf_matrix_tern.shape[0]):
            for j in range(conf_matrix_tern.shape[1]):
                plt.text(j, i, f'{conf_matrix_tern.iloc[i, j]:.2f}', ha='center', va='center', color='w', fontsize=12)

        # Add a color bar to indicate value-to-color mapping
        plt.colorbar(label='Intensity')

        # Set labels for the X and Y axes, if desired
        plt.xticks(range(conf_matrix_tern.shape[1]), conf_matrix_tern.columns)
        plt.yticks(range(conf_matrix_tern.shape[0]), conf_matrix_tern.index)

        # Show the plot
        plt.title("Relative Frequency Confusion Matrix")
        plt.show()
        
        # Adding  general title
        fig.suptitle(nome_matrix)
        
        # Saving the graph
        fig.savefig(f'{nome_matrix}.png')
    
        
    # Uniting the functions of testing and evaluating model performance '''
    
    def plotagem_geral(self, hiperparametros: str, dados: pd.DataFrame, nome_previsoes: str, nome_y: str, nome_tabela: str, nome_matrix: str):
    
        ''' Plotting the evaluations performed on the data '''
        
        # Evaluating the best model
        best_model = hiperparametros(dados)
        
        # Splitting the dataset into training (60%), testing (30%), and validation (10%)
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(dados, target = self.predicao, 
                                                                                train_size=0.60, valid_size=0.1, test_size=0.30)
        
        X_test = pd.concat([X_valid, X_test], axis=0)
        y_test = pd.concat([y_valid, y_test], axis=0)

        # Defining the prediction model
        modelo = best_model
        # Performing model training and testing
        modelo_treinado = modelo.fit(X_train, y_train)  
        previsoes = modelo_treinado.predict(X_test)

        # Saving predictions and y_test using the joblib library
        joblib.dump(previsoes, nome_previsoes)
        joblib.dump(y_test, nome_y)

        self.calculo_metricas(dados, best_model, nome_tabela)
        tabela_metricas = joblib.load(nome_tabela)

        # Evaluating precision, recall and f1-score
        print(classification_report(y_test, previsoes))
        
        # Building confusion matrix and relative frequency matrix
        if self.predicao == 'Risk stratification':
            print(self.matrix_confusao_ternaria(y_test, previsoes, nome_matrix))
        elif self.predicao == 'Exposure to pesticides':
            print(self.matrix_confusao(y_test, previsoes, nome_matrix))
        elif self.predicao == 'Worst prognosis':
            print(self.matrix_confusao(y_test, previsoes, nome_matrix))

        return tabela_metricas, y_test, previsoes


        ''' Functions to evaluate model performance '''
    
    def boxplot(self, dados_1: list, dados_2: list, dados_3: list, dados_4: list, titulo: list, margin_factor=0.1):
    
        ''' Structuring the boxplot to compare evaluation metrics between tested algorithms '''

        # Creating 2x2 subplots for boxplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Configuring the Seaborn style for subplots
        plt.style.use('seaborn-whitegrid')

        # Iterating over the data and creating the boxplots
        dados_boxplot = [dados_1, dados_2, dados_3, dados_4]

        # Concatenating all data to calculate global y-axis limits
        all_data = np.concatenate(dados_boxplot)

        # Finding the overall minimum and maximum values
        global_min = np.min(all_data)
        global_max = np.max(all_data)

        # Adding margin to the limits
        margin = (global_max - global_min) * margin_factor
        global_min -= margin
        global_max += margin

        for i, dados in enumerate(dados_boxplot):
            row, col = divmod(i, 2) 

            # Creating different colors for each box in the group
            cores_box = ['lightblue', 'pink', 'lightgreen', 'purple'][:len(dados[0])]

            # Adding the name of each machine learning model to the numeric indicator
            axs[row, col].set_xticklabels(['Logistic \n regression', 'Random \n Florest', 'SVM', 'GBoost'])

            # Creating boxplots with different colors for each box
            bp = axs[row, col].boxplot(dados, patch_artist=True)
            for patch, cor in zip(bp['boxes'], cores_box):
                patch.set_facecolor(cor)

            # Adding title to each boxplot
            axs[row, col].set_title(f'{titulo[i]}')

            # Setting y-axis limits based on overall minimum and maximum
            axs[row, col].set_ylim([global_min, global_max])

        # Adding labels and general title
        fig.suptitle(self.titulo_plot)
        for ax in axs.flat:
            ax.set_ylabel('Percentage (%)')

        # Adjusting the layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Displaying the graph
        plt.show()

        # Saving the graph
        fig.savefig(f'{self.titulo_plot}.png')
        
    def roc_curve_multiple_datasets(self, y_test, previsao, titulos: str):
    
        ''' Constructing ROC curves to compare the performance of tested algorithms '''

        # Create subplots based on the number of datasets
        n_datasets = len(y_test)

        # Create 3x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Setting Seaborn style for subplots
        plt.style.use('seaborn-whitegrid')

        # List of class names
        if self.predicao == 'Exposure to pesticides':
            class_names = ['Exposed'] 
        elif self.predicao == 'Worst prognosis':
            class_names = ['Worst prognosis']

        # Iterate over datasets
        for i, (y_test, previsao, titulo) in enumerate(zip(y_test, previsao, titulos)):

            # Determine which row and column the current plot should belong to
            row = i // 2
            col = i % 2
            
            # Create DataFrame for observed values
            df_predictions = pd.DataFrame(previsao)
            df_predictions.rename(columns={'Prediction': 0}, inplace=True)


            # Create DataFrame for observed values
            df_observed = pd.DataFrame(y_test)
            df_observed.rename(columns={'Exposure to pesticides': 0}, inplace=True)


            # Iterate over classes
            for class_label in range(df_observed.shape[1]):
                # Calculate false positive rate (FPR) and true positive rate (TPR)
                fpr, tpr, _ = roc_curve(np.array(df_observed)[:, class_label], np.array(df_predictions)[:, class_label])
                # Calculate AUC for the current class based on the previously calculated FPR and TPR values
                roc_auc = auc(fpr, tpr)

                # Plot ROC curves in each subplot
                axs[row, col].plot(fpr, tpr, label=f'{class_names[class_label]} (AUC = {roc_auc:.2f})')

            # Adding title to subplot
            axs[row, col].set_title(titulo)


        # Adding reference line for the first 5 subplots
        for i, ax in enumerate(axs.flatten()):
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.legend(loc='lower right')

        # Adding legends in each subplot
        for ax in axs.flat:
            ax.legend(loc='lower right')

        # Adding labels to axes
        for ax in axs.flat:
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')


        # Adding general title
        fig.suptitle(self.titulo_geral)

        # Adjust layout
        plt.tight_layout()

        # Display the image
        plt.show()

        # Saving the graph
        fig.savefig(f'{self.titulo_geral}.png')
        

    def roc_curve_estratifica(self,y_test, previsao, titulos):

            ''' Building ROC curves for multiple models '''

            # Create subplots based on the number of datasets
            n_datasets = len(y_test)

            # Create 3x2 subplots
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))

            # Setting Seaborn style for subplots
            plt.style.use('seaborn-whitegrid')

            # List of class names
            class_names = ['High risk', 'Intermediate risk', 'Low risk']

            # Iterate over datasets
            for i, (y_test, previsao, titulo) in enumerate(zip(y_test, previsao, titulos)):

                # Determine which row and column the current plot should belong to
                row = i // 2
                col = i % 2

                # Create DataFrame for observed values
                df_predictions = pd.DataFrame(previsao)
                df_predictions.rename(columns={'Prediction': 0}, inplace=True)

                # Apply one-hot encoding to predictions
                one_hot_encoded = pd.get_dummies(df_predictions)

                # Concatenate the result with the original DataFrame
                df_encoded_predictions = pd.concat([df_predictions, one_hot_encoded], axis=1)

                # Checking if column '0_Low' is present in predictions
                if '0_Low' not in df_encoded_predictions.columns:
                    # Adding a '0_Low' column with zero values
                    df_encoded_predictions['0_Low'] = 0

                # Checking if column '0_Medium' is present in predictions
                if '0_Medium' not in df_encoded_predictions.columns:
                    # Adding a '0_Low' column with zero values
                    df_encoded_predictions['0_Medium'] = 0

                # Checking if column '0_High' is present in predictions
                if '0_High' not in df_encoded_predictions.columns:
                    # Adding a '0_Low' column with zero values
                    df_encoded_predictions['0_High'] = 0

                df_encoded_predictions = df_encoded_predictions[['0_High', '0_Medium', '0_Low']]

                # Create DataFrame for observed values
                df_observed = pd.DataFrame(y_test)
                df_observed.rename(columns={'Risk stratification': 0}, inplace=True)

                # Apply one-hot encoding
                one_hot_encoded = pd.get_dummies(df_observed)

                # Concatenate the result with the original DataFrame
                df_encoded_observed = pd.concat([df_observed, one_hot_encoded], axis=1)

                # Checking if column '0_Low' is present in predictions
                if '0_Low' not in df_encoded_observed.columns:
                    # Adding a '0_Low' column with zero values
                    df_encoded_observed['0_Low'] = 0

                df_encoded_observed = df_encoded_observed[['0_High', '0_Medium', '0_Low']]

                # Iterate over classes
                for class_label in range(df_encoded_observed.shape[1]):
                    # Calculate false positive rate (FPR) and true positive rate (TPR)
                    fpr, tpr, _ = roc_curve(np.array(df_encoded_observed)[:, class_label],
                                            np.array(df_encoded_predictions)[:, class_label])
                    # Calculate AUC for the current class based on the previously calculated FPR and TPR values
                    roc_auc = auc(fpr, tpr)

                    # Plot ROC curves in each subplot
                    axs[row, col].plot(fpr, tpr, label=f'{class_names[class_label]} (AUC = {roc_auc:.2f})')

                # Adding title to subplot
                axs[row, col].set_title(titulo)


            # Adding reference line for the first 5 subplots
            for i, ax in enumerate(axs.flatten()):
                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.legend(loc='lower right')

            # Adding legends in each subplot
            for ax in axs.flat:
                ax.legend(loc='lower right')

            # Adding labels to axes
            for ax in axs.flat:
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')

            # Adding general title
            fig.suptitle(self.titulo_geral)

            # Adjust layout
            plt.tight_layout()

            # Display the image
            plt.show()

            # Saving the graph
            fig.savefig(f'{self.titulo_geral}.png')
        
    
    def armazena_nome_arquivos(self, dados, svmm, rll, rff, gb):
        
            
        svm_2 = svmm
        rl_2 = rll
        rf_2 = rff
        gboost_2 = gb
    

        svm = [self.grid_search_SVM, dados]
        rl = [self.grid_search_RL, dados]
        rf = [self.grid_search_RF, dados]
        gboost = [self.grid_search_GBOOST, dados]

        lista_svm = svm + svm_2
        lista_rl = rl + rl_2
        lista_rf = rf + rf_2
        lista_gboost = gboost + gboost_2
        

        return lista_svm, lista_rl, lista_rf, lista_gboost

        
    def prediction(self, svmm, rll, rff, gb):
        
                
        data = self.estrutura_dados()
        
        informacoes_dataframe = self.informacoes_df(data)
        
        svm, rl, rf, gboost = self.armazena_nome_arquivos(data, svmm, rll, rff, gb)
        
        tabela_svm, y_test_svm, previsoes_svm = self.plotagem_geral(svm[0], svm[1], svm[2], svm[3], svm[4], svm[5])
        tabela_rl, y_test_rl, previsoes_rl = self.plotagem_geral(rl[0], rl[1], rl[2], rl[3], rl[4], rl[5])
        tabela_gboost, y_test_gboost, previsoes_gboost =self.plotagem_geral(gboost[0], gboost[1], gboost[2], gboost[3], gboost[4], gboost[5])
        tabela_rf, y_test_rf, previsoes_rf = self.plotagem_geral(rf[0], rf[1], rf[2], rf[3], rf[4], rf[5])
        

        rf = tabela_rf
        regressao = tabela_rl
        svm = tabela_svm
        gboost = tabela_gboost
       

        plot_acuracia = [regressao['Accuracy'].tolist(), rf['Accuracy'].tolist(), svm['Accuracy'].tolist(), gboost['Accuracy'].tolist()]
        plot_precisao = [regressao['Precision'].tolist(), rf['Precision'].tolist(), svm['Precision'].tolist(),  gboost['Precision'].tolist()]
        plot_recall = [regressao['Recall'].tolist(), rf['Recall'].tolist(), svm['Recall'].tolist(),  gboost['Recall'].tolist()]
        plot_f1 = [regressao['F1-score'].tolist(), rf['F1-score'].tolist(), svm['F1-score'].tolist(), gboost['F1-score'].tolist()]
        
        

        plot = self.boxplot(plot_acuracia, plot_precisao, plot_recall, plot_f1, ['Accuracy', 'Precision', 'Recall', 'F1-score'])

        
        if self.predicao == 'Exposure to pesticides':
            curva_roc = self.roc_curve_multiple_datasets([y_test_rl, y_test_rf, y_test_svm, y_test_gboost],
                                                         [previsoes_rl, previsoes_rf, previsoes_svm, previsoes_gboost], 
                                                         ['ROC Curves - Logistic regression', "ROC Curves - Random Forest", 'ROC Curves - SVM','ROC Curves - GBoost'])
            
        elif self.predicao == 'Worst prognosis':
            curva_roc = self.roc_curve_multiple_datasets([y_test_rl, y_test_rf, y_test_svm, y_test_gboost],
                                                         [previsoes_rl, previsoes_rf, previsoes_svm, previsoes_gboost], 
                                                         ['ROC Curves - Logistic regression', "ROC Curves - Random Forest", 'ROC Curves - SVM','ROC Curves - GBoost'])
        else: 
            curva_roc = self.roc_curve_estratifica([y_test_rl, y_test_rf, y_test_svm, y_test_gboost],
                                                         [previsoes_rl, previsoes_rf, previsoes_svm, previsoes_gboost], 
                                                         ['ROC Curves - Logistic regression', "ROC Curves - Random Forest", 'ROC Curves - SVM','ROC Curves - GBoost'])
            

        return plot, curva_roc