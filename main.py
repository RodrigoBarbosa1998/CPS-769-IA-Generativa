import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class Main:
    def __init__(self):
        self.data_path = self.get_data_path()
        self.students_data_new = None
        self.students_data = None
        self.model = None
        self.scaler = None

    def get_data_path(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_path, 'data')
        return data_path

    def load_data(self):
        try:
            data_new_path = os.path.join(self.data_path, 'lista2-students_data_new.ods')
            data_path = os.path.join(self.data_path, 'lista2-students_data.ods')
            
            self.students_data_new = self.read_ods(data_new_path)
            self.students_data = self.read_ods(data_path)
        except ImportError as e:
            print(f"Erro ao importar dependência: {e}")
            raise

    def read_ods(self, file_path):
        try:
            from odf.opendocument import load
            from odf.table import Table, TableRow, TableCell
            from odf.text import P
        except ImportError as e:
            print(f"Erro ao importar dependência: {e}")
            raise

        doc = load(file_path)
        sheet = doc.spreadsheet.getElementsByType(Table)[0]
        data = []

        for row in sheet.getElementsByType(TableRow):
            row_data = []
            for cell in row.getElementsByType(TableCell):
                value = cell.getElementsByType(P)[0].firstChild.data
                row_data.append(value)
            data.append(row_data)

        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Converter as colunas 'Presença', 'HorasEstudo' e 'Nota' para float, substituindo vírgulas por pontos
        df['Presença'] = df['Presença'].str.replace(',', '.').astype(float)
        df['HorasEstudo'] = df['HorasEstudo'].str.replace(',', '.').astype(float)
        if 'Nota' in df.columns:
            df['Nota'] = df['Nota'].str.replace(',', '.').astype(float)
        
        return df

    def preprocess_data(self, df):
        df['HorasEstudo'] = df['HorasEstudo'].apply(lambda x: min(x, 8))
        return df

    def train_model(self):
        if 'Nota' not in self.students_data.columns:
            print("Erro: A coluna 'Nota' está ausente nos dados de treinamento.")
            return
        
        self.students_data = self.preprocess_data(self.students_data)

        X = self.students_data[['Presença', 'HorasEstudo']].astype(float)
        y = self.students_data['Nota'].astype(float)

        # Dividir os dados em 80% para treino e 20% para teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Escalonar os dados
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Configurar e treinar o modelo
        self.model = MLPRegressor(hidden_layer_sizes=(3,), activation='relu', solver='adam', max_iter=1000, learning_rate_init=0.01, verbose=True, random_state=1)
        self.model.fit(X_train_scaled, y_train)

        # Fazer previsões e calcular o erro quadrático médio
        predictions = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)

        print(f"Erro Quadrático Médio: {mse}")
        print(f"Pesos da camada oculta: {self.model.coefs_}")
        print(f"Bias da camada oculta: {self.model.intercepts_}")
        print(f"Pesos da camada de saída: {self.model.coefs_[-1]}")
        print(f"Bias da camada de saída: {self.model.intercepts_[-1]}")

    def predict(self):
        if self.model is None:
            print("Erro: O modelo não foi treinado.")
            return
        
        if self.students_data_new is None:
            print("Erro: Dados novos não foram carregados.")
            return
        
        self.students_data_new = self.preprocess_data(self.students_data_new)
        X_new = self.students_data_new[['Presença', 'HorasEstudo']].astype(float)
        X_new_scaled = self.scaler.transform(X_new)
        predictions = self.model.predict(X_new_scaled)
        
        # Adicionar as previsões ao DataFrame de dados novos
        self.students_data_new['Nota'] = predictions
        
        print("Previsões para 'Nota':")
        print(self.students_data_new.head())
        
        # Opcional: Salvar os resultados em um novo arquivo
        output_path = os.path.join(self.data_path, 'students_data_predictions.ods')
        self.save_to_ods(self.students_data_new, output_path)
        print(f"Previsões salvas em: {output_path}")

    def save_to_ods(self, df, file_path):
        from odf.opendocument import OpenDocumentSpreadsheet
        from odf.table import Table, TableRow, TableCell
        from odf.text import P

        doc = OpenDocumentSpreadsheet()
        table = Table(name="Sheet1")
        doc.spreadsheet.addElement(table)
        
        # Add header row
        header_row = TableRow()
        for column in df.columns:
            cell = TableCell()
            cell.addElement(P(text=column))
            header_row.addElement(cell)
        table.addElement(header_row)
        
        # Add data rows
        for _, row in df.iterrows():
            data_row = TableRow()
            for value in row:
                cell = TableCell()
                cell.addElement(P(text=str(value)))
                data_row.addElement(cell)
            table.addElement(data_row)
        
        doc.save(file_path)

    def run(self):
        self.load_data()
        self.train_model()
        self.predict()

if __name__ == "__main__":
    main = Main()
    main.run()
